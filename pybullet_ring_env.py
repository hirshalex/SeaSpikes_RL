# pybullet_ring_env.py
"""
RingPickPlaceEnv (PyBullet)
- Multi-tentacle (Sea-Spikes style) pick-and-place environment.
- Parametric: number of tentacles, tentacle curvature, base radius, max tentacles.
- Each tentacle gets a matching colored torus (ring) spawned near the robot/gripper.
- Rings spawn in a spawn zone near the robot; tentacles arranged in a circle.
- Designed to scale: default is 1 tentacle (easy). Increase to up to max_tentacles (10).
"""
import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import gymnasium as gym

# ---------- Utility: write a torus OBJ (used to create a real ring/torus) ----------
def write_torus_obj(path, R=0.035, r=0.01, n_major=48, n_minor=20):
    """
    Write a torus mesh (.obj) to 'path'. R = major radius, r = minor radius.
    Resolution: n_major x n_minor
    """
    verts = []
    faces = []
    for i in range(n_major):
        theta = (i / n_major) * 2.0 * math.pi
        for j in range(n_minor):
            phi = (j / n_minor) * 2.0 * math.pi
            x = (R + r * math.cos(phi)) * math.cos(theta)
            y = (R + r * math.cos(phi)) * math.sin(theta)
            z = r * math.sin(phi)
            verts.append((x, y, z))
    for i in range(n_major):
        for j in range(n_minor):
            i1 = i
            j1 = j
            i2 = (i + 1) % n_major
            j2 = (j + 1) % n_minor
            v0 = i1 * n_minor + j1
            v1 = i2 * n_minor + j1
            v2 = i2 * n_minor + j2
            v3 = i1 * n_minor + j2
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))


class RingPickPlaceEnv(gym.Env):
    """
    Gym-style PyBullet environment.
    Configurable params: num_tentacles, max_tentacles, tentacle_curve, base_radius, etc.
    Observation: joint angles (num_joints), joint vel, ee pos, ee orn, for each tentacle: tentacle top pos (x,y,z) and ring pos (x,y,z) and color idx (int).
    Action: [dx, dy, dz, grip] - end-effector delta in meters (scaled) + grip open/close.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 gui=False,
                 timestep=1/240.,
                 frame_skip=4,
                 num_tentacles=1,
                 max_tentacles=10,
                 base_circle_radius=0.18,
                 tentacle_segments=8,
                 tentacle_segment_length=0.035,
                 tentacle_curve=0.0,
                 color_palette=None):
        super().__init__()

        # simulation / display
        self.gui = gui
        self.timestep = timestep
        self.frame_skip = frame_skip

        # environment layout parameters (scalable)
        self.num_tentacles = int(num_tentacles)
        self.max_tentacles = int(max_tentacles)
        self.base_circle_radius = float(base_circle_radius)
        self.tentacle_segments = int(tentacle_segments)
        self.tentacle_segment_length = float(tentacle_segment_length)
        self.tentacle_curve = float(tentacle_curve)  # radians per segment for curvature

        # color palette (8 default colors)
        if color_palette is None:
            # RGBA palette (8 colors)
            self.color_palette = [
                [1.0, 0.2, 0.2, 1.0],   # red
                [0.2, 1.0, 0.2, 1.0],   # green
                [0.2, 0.4, 1.0, 1.0],   # blue
                [1.0, 0.7, 0.2, 1.0],   # orange
                [0.8, 0.2, 1.0, 1.0],   # magenta
                [0.2, 1.0, 1.0, 1.0],   # cyan
                [0.9, 0.9, 0.2, 1.0],   # yellow
                [0.6, 0.3, 0.9, 1.0],   # purple-ish
            ]
        else:
            self.color_palette = color_palette

        # workspace bounds (x,y,z)
        self.workspace_low = np.array([0.2, -0.45, 0.0])
        self.workspace_high = np.array([0.8, 0.45, 0.5])

        # physics connect
        if self.gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)

        # bookkeeping for bodies
        self.tentacle_ids = []          # list of body ids for current tentacle segments (flat list)
        self.tentacle_top_positions = []# list of 3-d positions (one per tentacle)
        self.tentacle_color_idxs = []   # list of color indices for each tentacle
        self.ring_ids = []              # list of ring body ids (one per tentacle)
        self.spawn_zone_center = np.array([0.35, -0.35, 0.06])  # where rings spawn (near gripper)
        self.spawn_zone_size = np.array([0.1, 0.2, 0.02])      # region dims for ring spawn
        
        # bookkeeping for reward / grip tracking
        self._last_grip = 0.0        # track previous grip value (for toggle penalty)
        self._hold_reward_per_step = 0.02  # small per-step reward while holding
        self._toggle_penalty = 0.05        # small penalty when grip sign toggles quickly


        # agent / gripper state
        self.holding = False
        self.hold_constraint = None
        self.held_ring_id = None

        # control params
        self.max_delta = 0.06  # max meters per action step
        self.grasp_distance = 0.12
        self.place_distance = 0.06
        self.success_reward = 5.0

        # build scene and robot
        self._load_scene()

        # action/observation spaces
        # Action: dx,dy,dz, grip
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation:
        # joint angles (num_joints) + joint vels (num_joints) + ee_pos(3) + ee_orn(4)
        # + for each tentacle up to max_tentacles: tentacle_top_pos(3), ring_pos(3), color_idx (1)
        num_joints = self.num_joints
        obs_dim = num_joints + num_joints + 3 + 4 + self.max_tentacles * (3 + 3 + 1)
        obs_high = np.ones(obs_dim, dtype=np.float32) * 10.0   # finite upper bound
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)


        # miscellaneous
        self.seed()
        self._ensure_torus_mesh()

    # ---------------- utilities ----------------
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _ensure_torus_mesh(self):
        mesh_path = os.path.join(os.path.dirname(__file__), "torus.obj")
        if not os.path.exists(mesh_path):
            write_torus_obj(mesh_path, R=0.035, r=0.010, n_major=48, n_minor=20)
        self._torus_path = mesh_path

    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # plane
        p.loadURDF("plane.urdf")

        # load robot (KUKA)
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.kuka = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True, flags=flags)
        self.num_joints = p.getNumJoints(self.kuka)
        # try to find end-effector link index
        self.ee_link_index = None
        for i in range(self.num_joints):
            info = p.getJointInfo(self.kuka, i)
            name = info[12].decode('utf-8')
            if "lbr_iiwa_link_7" in name or "link_7" in name or "ee_link" in name:
                self.ee_link_index = i
        if self.ee_link_index is None:
            self.ee_link_index = self.num_joints - 1

        # set neutral joint states
        for j in range(self.num_joints):
            p.resetJointState(self.kuka, j, targetValue=0.0, targetVelocity=0.0)

        # clear any previous lists
        self.tentacle_ids = []
        self.tentacle_top_positions = []
        self.tentacle_color_idxs = []
        self.ring_ids = []
        self.holding = False
        self.hold_constraint = None
        self.held_ring_id = None

    # ---------------- tentacle / ring spawning ----------------
    def _clear_existing_bodies(self):
        # remove previous ring bodies
        for rid in list(self.ring_ids):
            try:
                p.removeBody(rid)
            except Exception:
                pass
        self.ring_ids = []
        # remove previous tentacle bodies
        for tid in list(self.tentacle_ids):
            try:
                p.removeBody(tid)
            except Exception:
                pass
        self.tentacle_ids = []
        self.tentacle_top_positions = []
        self.tentacle_color_idxs = []
        # detach held constraint
        if self.hold_constraint is not None:
            try:
                p.removeConstraint(self.hold_constraint)
            except Exception:
                pass
            self.hold_constraint = None
            self.holding = False
            self.held_ring_id = None

    def _arrange_tentacle_positions(self, n):
        """
        Arrange n tentacles evenly around a circle centered roughly in front of robot.
        Returns list of (x,y,z, yaw) positions.
        """
        cx = (self.workspace_low[0] + self.workspace_high[0]) / 2.0
        cy = 0.0
        base_radius = max(self.base_circle_radius, 0.05 + 0.02 * n)  # scale with n slightly
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        poses = []
        for a in angles:
            x = cx + base_radius * math.cos(a)
            y = cy + base_radius * math.sin(a)
            z = 0.0
            yaw = math.degrees(a) + 90.0  # orientation for visual placement
            poses.append((x, y, z, yaw))
        return poses

    def _create_tentacle(self, base_pos, color_rgba, segments=None, seg_len=None, base_radius=0.045, tip_radius=0.012, curve=0.0, axis='y'):
        """
        Create a tapered tentacle made of cylinder segments connected by fixed constraints.
        Returns (list_of_body_ids, top_position)
        """
        if segments is None:
            segments = self.tentacle_segments
        if seg_len is None:
            seg_len = self.tentacle_segment_length

        body_ids = []
        x0, y0, z0 = base_pos
        radii = np.linspace(base_radius, tip_radius, segments)

        parent_id = None
        for i in range(segments):
            r = float(radii[i])
            length = float(seg_len)
            # collision & visual
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=length)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=length, rgbaColor=color_rgba)
            # orientation (apply curvature by small incremental rotations)
            if i == 0:
                orient = p.getQuaternionFromEuler([0, 0, 0])
            else:
                angle = curve * i
                if axis == 'y':
                    orient = p.getQuaternionFromEuler([0, angle, 0])
                else:
                    orient = p.getQuaternionFromEuler([angle, 0, 0])
            # compute a simple approximate segment center position: stack along Z and add lateral offset if curving
            lateral_offset = math.sin(curve * i) * length * i if curve != 0.0 else 0.0
            if axis == 'y':
                seg_pos = [x0 + lateral_offset, y0, z0 + (i + 0.5) * length]
            else:
                seg_pos = [x0, y0 + lateral_offset, z0 + (i + 0.5) * length]

            body_id = p.createMultiBody(baseMass=0.0,
                                        baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis,
                                        basePosition=seg_pos,
                                        baseOrientation=orient)
            body_ids.append(body_id)

            if parent_id is not None:
                # fixed constraint linking the bottom of current to top of parent
                p.createConstraint(parentBodyUniqueId=parent_id,
                                   parentLinkIndex=-1,
                                   childBodyUniqueId=body_id,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_FIXED,
                                   jointAxis=[0,0,0],
                                   parentFramePosition=[0, 0, length/2.0],
                                   childFramePosition=[0, 0, -length/2.0])
            parent_id = body_id

        top_z = z0 + segments * seg_len
        top_pos = np.array([x0, y0, top_z])
        return body_ids, top_pos

    def _spawn_ring_for_color(self, spawn_center, color_rgba):
        """
        Spawn a torus ring with given color within spawn_zone.
        Returns body id and position.
        """
        # sample a random offset in spawn zone
        offs = np.array([self.np_random.uniform(-0.5, 0.5) * self.spawn_zone_size[0],
                         self.np_random.uniform(-0.5, 0.5) * self.spawn_zone_size[1],
                         0.0])
        pos = spawn_center + offs
        pos[2] = max(pos[2], 0.04)
        # create visual and collision (mesh)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=self._torus_path, meshScale=[1,1,1])
        vis = p.createVisualShape(p.GEOM_MESH, fileName=self._torus_path, meshScale=[1,1,1], rgbaColor=color_rgba)
        body = p.createMultiBody(baseMass=0.05,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=pos.tolist())
        return body, pos

    # ---------------- stepping / grasping ----------------
    def _get_ee_pose(self):
        st = p.getLinkState(self.kuka, self.ee_link_index, computeForwardKinematics=True)
        pos = np.array(st[4])
        orn = np.array(st[5])  # quat
        return pos, orn

    def _attach_ring(self, ring_id):
        if self.hold_constraint is None and ring_id is not None:
            # attach ring rigidly to EE link
            cid = p.createConstraint(parentBodyUniqueId=self.kuka,
                                     parentLinkIndex=self.ee_link_index,
                                     childBodyUniqueId=ring_id,
                                     childLinkIndex=-1,
                                     jointType=p.JOINT_FIXED,
                                     jointAxis=[0,0,0],
                                     parentFramePosition=[0,0,0],
                                     childFramePosition=[0,0,0])
            self.hold_constraint = cid
            self.holding = True
            self.held_ring_id = ring_id

    def _release_ring(self):
        if self.hold_constraint is not None:
            try:
                p.removeConstraint(self.hold_constraint)
            except Exception:
                pass
            self.hold_constraint = None
        self.holding = False
        self.held_ring_id = None

    # ---------------- main Gym API ----------------
    def step(self, action):
        """
        FINAL COMPREHENSIVE REWARD FUNCTION
        
        Key improvements:
        1. Explicit gripper state rewards (encourages correct open/close at right times)
        2. Penalty for wrong gripper actions (discourages spam)
        3. Bonus for maintaining grasp (prevents dropping)
        4. Curriculum-friendly: works for both easy and hard scenarios
        """
        # Initialize trackers
        if not hasattr(self, "step_count"):
            self.step_count = 0
        if not hasattr(self, "was_holding_last_step"):
            self.was_holding_last_step = False
        if not hasattr(self, "prev_nearest_d"):
            self.prev_nearest_d = None
        if not hasattr(self, "prev_ring_to_target_d"):
            self.prev_ring_to_target_d = None
        if not hasattr(self, "last_grip_action"):
            self.last_grip_action = 0.0
        if not hasattr(self, "steps_since_grasp"):
            self.steps_since_grasp = 0
        
        self.step_count += 1

        # Parse and clip action
        action = np.clip(action, -1.0, 1.0)
        dx, dy, dz, grip_action = action
        delta = np.array([dx, dy, dz], dtype=np.float32) * self.max_delta

        # Move end effector via IK
        ee_pos, ee_orn = self._get_ee_pose()
        target_pos = ee_pos + delta
        target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)

        target_joints = p.calculateInverseKinematics(
            self.kuka, self.ee_link_index, target_pos, maxNumIterations=20
        )
        for j in range(len(target_joints)):
            p.setJointMotorControl2(
                bodyUniqueId=self.kuka,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[j],
                force=200
            )

        for _ in range(self.frame_skip):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.timestep)

        ee_pos, ee_orn = self._get_ee_pose()

        # Find nearest ring
        nearest_ring = None
        nearest_d = 1e6
        for rid in self.ring_ids:
            try:
                rpos, _ = p.getBasePositionAndOrientation(rid)
                d = np.linalg.norm(np.array(rpos) - ee_pos)
                if d < nearest_d:
                    nearest_d = d
                    nearest_ring = rid
            except:
                continue

        if self.prev_nearest_d is None:
            self.prev_nearest_d = nearest_d

        # ============================================
        # GRIPPER LOGIC (unchanged)
        # ============================================
        grasped_this_step = False
        released_this_step = False
        
        if grip_action > 0.5:
            if not self.holding and nearest_ring is not None and nearest_d < self.grasp_distance:
                self._attach_ring(nearest_ring)
                grasped_this_step = True
                self.steps_since_grasp = 0
        elif grip_action < -0.5:
            if self.holding:
                self._release_ring()
                released_this_step = True

        # ============================================
        # REWARD FUNCTION - COMPREHENSIVE
        # ============================================
        reward = 0.0
        done = False
        info = {}

        # Tiny time penalty
        reward -= 0.001

        if not self.holding:
            # ============================================
            # PHASE 1: Approach and Grasp
            # ============================================
            
            # 1. Delta reward for getting closer (MOST IMPORTANT)
            delta_distance = self.prev_nearest_d - nearest_d
            reward += 5.0 * delta_distance  # Increased from 2.0 to dominate
            
            # 2. Distance shaping - NEGATIVE when far, positive when close
            # Only positive within ~0.3m, negative beyond that
            if nearest_d < 0.3:
                # Close enough - positive exponential reward
                distance_reward = np.exp(-5.0 * nearest_d)
                reward += 1.0 * distance_reward
            else:
                # Too far - linear negative penalty
                reward += -0.5 * (nearest_d - 0.3)
            
            # 3. Milestone bonus for entering grasp zone
            if nearest_d < self.grasp_distance * 1.5:
                reward += 1.0
            
            # 4. GRIPPER BEHAVIOR SHAPING
            # When close to ring, reward CLOSING gripper
            # When far from ring, slight penalty for closing (discourages spam)
            if nearest_d < self.grasp_distance * 1.2:
                # In grasp range - reward closing attempts
                if grip_action > 0.5:
                    reward += 0.5  # "Good! Try to grasp when close!"
            else:
                # Far from ring - small penalty for trying to grasp
                if grip_action > 0.5:
                    reward -= 0.2  # "Don't grasp when far away"
            
            # 5. HUGE bonus for successful grasp
            if grasped_this_step:
                reward += 25.0
                info['grasped'] = True
        
        else:
            # ============================================
            # PHASE 2: Transport and Place
            # ============================================
            
            self.steps_since_grasp += 1
            
            try:
                held_idx = self.ring_ids.index(self.held_ring_id)
                target_pos = self.tentacle_top_positions[held_idx]
                ring_pos, _ = p.getBasePositionAndOrientation(self.held_ring_id)
                ring_pos = np.array(ring_pos)
                target_pos = np.array(target_pos)
                
                dist_to_target = np.linalg.norm(ring_pos - target_pos)
                
                if self.prev_ring_to_target_d is None:
                    self.prev_ring_to_target_d = dist_to_target
                
                # 6. Delta reward for approaching target
                delta_to_target = self.prev_ring_to_target_d - dist_to_target
                reward += 3.0 * delta_to_target
                
                # 7. Exponential shaping toward target
                target_reward = np.exp(-3.0 * dist_to_target)
                reward += 3.0 * target_reward
                
                # 8. HOLDING BONUS (NEW!)
                # Reward for keeping the ring (prevents dropping)
                reward += 0.1  # Per-step bonus for maintaining grasp
                
                # 9. GRIPPER BEHAVIOR WHILE HOLDING (NEW!)
                # Strongly penalize trying to OPEN gripper unless at target
                if dist_to_target > self.place_distance * 2.0:
                    # Still transporting - heavily penalize opening
                    if grip_action < -0.5:
                        reward -= 2.0  # "Don't drop it!"
                else:
                    # Near target - reward opening (to place)
                    if grip_action < -0.5:
                        reward += 1.0  # "Good! Release when at target!"
                
                # 10. SUCCESS - placed at target
                if dist_to_target < self.place_distance:
                    reward += 100.0
                    done = True
                    info['success'] = True
                
                self.prev_ring_to_target_d = dist_to_target
                
            except (ValueError, IndexError):
                reward -= 1.0
        
        # 11. PENALTY FOR LOSING GRASP (NEW!)
        # If we were holding last step but not this step, and we didn't intentionally release
        if self.was_holding_last_step and not self.holding and not released_this_step:
            # Lost grasp accidentally (ring fell or flew away)
            reward -= 10.0  # Strong penalty for dropping
            info['dropped'] = True

        # Update trackers
        self.was_holding_last_step = self.holding
        self.prev_nearest_d = nearest_d
        self.last_grip_action = grip_action

        obs = self._get_obs()
        return obs.astype(np.float32), float(reward), done, False, info


    # ============================================
    # COMPREHENSIVE REWARD STRUCTURE
    # ============================================
    # 
    # PHASE 1 - Not Holding (Approach & Grasp):
    #   -0.001           time penalty
    #   +5.0 * Δdist     STRONG delta reward (getting closer)
    #   IF distance < 0.3m:
    #     +1.0 * exp(-5d)  positive exponential (only when close)
    #   ELSE:
    #     -0.5 * (d-0.3)   negative linear (penalty for being far)
    #   +1.0             milestone (entering grasp zone)
    #   +0.5             bonus for closing grip when close
    #   -0.2             penalty for closing grip when far
    #   +25.0            HUGE bonus for successful grasp
    #   -10.0            penalty for losing grasp accidentally
    #
    # PHASE 2 - Holding (Transport & Place):
    #   -0.001           time penalty
    #   +0.1             per-step holding bonus
    #   +3.0 * Δdist     delta toward target
    #   +3.0 * exp(-3d)  exponential shaping toward target
    #   -2.0             penalty for opening grip during transport
    #   +1.0             bonus for opening grip at target
    #   +100.0           MASSIVE success bonus
    #   -10.0            penalty for dropping ring
    #
    # KEY FIX:
    # - Distance shaping is NEGATIVE when far (>0.3m)
    # - Delta reward increased to 5.0 (dominates over shaping)
    # - No more "do nothing far away" local optimum 



    def _get_obs(self):
        # joint angles & velocities
        angles = []
        vels = []
        for j in range(self.num_joints):
            st = p.getJointState(self.kuka, j)
            angles.append(st[0])
            vels.append(st[1])
        angles = np.array(angles, dtype=np.float32)
        vels = np.array(vels, dtype=np.float32)

        ee_pos, ee_orn = self._get_ee_pose()

        # for each tentacle slot up to max_tentacles, provide: top_pos(3), ring_pos(3), color_idx(1)
        slots = []
        for i in range(self.max_tentacles):
            if i < len(self.tentacle_top_positions):
                tpos = np.array(self.tentacle_top_positions[i], dtype=np.float32)
            else:
                tpos = np.zeros(3, dtype=np.float32)
            if i < len(self.ring_ids):
                rpos, _ = p.getBasePositionAndOrientation(self.ring_ids[i])
                rpos = np.array(rpos, dtype=np.float32)
            else:
                rpos = np.zeros(3, dtype=np.float32)
            if i < len(self.tentacle_color_idxs):
                color_idx = float(self.tentacle_color_idxs[i])
            else:
                color_idx = -1.0
            slots.append(np.concatenate([tpos, rpos, np.array([color_idx], dtype=np.float32)]))
        slots = np.concatenate(slots).astype(np.float32)

        # compute nearest_d (include z)
        nearest_d = 1e6
        for rid in self.ring_ids:
            try:
                rpos, _ = p.getBasePositionAndOrientation(rid)
            except Exception:
                continue
            d = np.linalg.norm(np.array(rpos) - ee_pos)
            if d < nearest_d:
                nearest_d = d
        if nearest_d == 1e6:
            nearest_d = 0.0
        # then append nearest_d and holding flag:
        extra = np.array([nearest_d, 1.0 if self.holding else 0.0], dtype=np.float32)
        # and include `extra` into the final obs concat.


        obs = np.concatenate([angles, vels, ee_pos.astype(np.float32), ee_orn.astype(np.float32), slots])
        return obs

    def reset(self, num_tentacles=None, seed=None, options=None):
        """
        Reset the environment.

        Accepts `seed` and `options` for compatibility with newer Gym/Gymnasium / Stable-Baselines3 APIs.
        Returns the observation (same as before) for backwards compatibility with existing code.
        """
        # If a seed is provided by the caller (VecEnv/Stable-Baselines3), use it.
        if seed is not None:
            try:
                # prefer new-style RNG if available
                self.seed(seed)
            except Exception:
                # fallback (shouldn't occur) — ignore if seed can't be set
                pass

        # clear previous
        self._clear_existing_bodies()

        if num_tentacles is None:
            num_tentacles = self.num_tentacles
        else:
            # ensure valid integer within allowed bounds
            num_tentacles = int(max(1, min(self.max_tentacles, int(num_tentacles))))
        self.num_tentacles = num_tentacles

        # reposition robot to neutral start
        for j in range(self.num_joints):
            p.resetJointState(self.kuka, j, targetValue=0.0, targetVelocity=0.0)

        # arrange tentacle positions in a circle in front of the robot
        poses = self._arrange_tentacle_positions(num_tentacles)

        # choose colors in order (no immediate repeats until palette exhausted)
        palette = self.color_palette
        palette_len = len(palette)
        # start index random but deterministic per-episode to avoid repeating same color sequence each reset
        # Choose random start color index
        try:
            start_idx = int(self.np_random.integers(0, palette_len))
        except AttributeError:
            # fallback for older Gym seeding API returning RandomState
            start_idx = int(self.np_random.randint(0, palette_len))

        color_idxs = [(start_idx + i) % palette_len for i in range(num_tentacles)]

        # create tentacles and matching rings
        self.tentacle_top_positions = []
        self.tentacle_color_idxs = []
        self.ring_ids = []
        for i, pose in enumerate(poses):
            x, y, z, yaw = pose
            cid = color_idxs[i]
            color = palette[cid]
            # create tentacle
            t_ids, top_pos = self._create_tentacle([x, y, z], color_rgba=color,
                                                   segments=self.tentacle_segments,
                                                   seg_len=self.tentacle_segment_length,
                                                   base_radius=0.045,
                                                   tip_radius=0.012,
                                                   curve=self.tentacle_curve,
                                                   axis='y')
            self.tentacle_ids += t_ids
            self.tentacle_top_positions.append(top_pos)
            self.tentacle_color_idxs.append(cid)

        # spawn rings near spawn zone (one per tentacle), in color-matching order
        # spread them in spawn region so they aren't overlapping
        for i, cid in enumerate(color_idxs):
            color = palette[cid]
            body, pos = self._spawn_ring_for_color(self.spawn_zone_center, color)
            self.ring_ids.append(body)

        # optionally set a good camera (and make GUI interactive still)
        if self.gui:
            # focus camera on the center of the tentacle cluster
            cluster_center = np.array([ (self.workspace_low[0] + self.workspace_high[0]) / 2.0, 0.0, 0.08])
            self._set_camera(distance=0.45, yaw=60, pitch=-30, target=cluster_center.tolist())

        # ensure no hold
        self._release_ring()

        return self._get_obs().astype(np.float32), {}

    # ---------------- camera ----------------
    def _set_camera(self, distance=0.8, yaw=60, pitch=-30, target=None):
        if target is None:
            target = [(self.workspace_low[0] + self.workspace_high[0]) / 2.0, 0.0, 0.1]
        p.resetDebugVisualizerCamera(cameraDistance=distance,
                                     cameraYaw=yaw,
                                     cameraPitch=pitch,
                                     cameraTargetPosition=target)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            cam_target = [(self.workspace_low[0] + self.workspace_high[0]) / 2.0, 0, 0.1]
            cam_pos = [0.5, -1.0, 0.8]
            view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
                                              cameraTargetPosition=cam_target,
                                              cameraUpVector=[0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=5.0)
            w, h = 320, 320
            img = p.getCameraImage(width=w, height=h, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
            rgb = np.reshape(img[2], (h, w, 4))[:, :, :3]
            return rgb
        elif mode == 'human':
            return None

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass


# If executed directly, run a short GUI test
if __name__ == "__main__":
    env = RingPickPlaceEnv(gui=True, num_tentacles=1)
    env.reset()
    for _ in range(400):
        a = env.action_space.sample()
        obs, r, d, info = env.step(a)
        if d:
            env.reset()
    env.close()
