"""
RingPickPlaceEnv (PyBullet)
- Multi-tentacle (Sea-Spikes style) pick-and-place environment.
- Parametric: number of tentacles, tentacle curvature, base radius, max tentacles.
- Each tentacle gets a matching colored torus (ring) spawned near the robot/gripper.
- Rings spawn in a spawn zone near the robot; tentacles arranged in a circle.
- Designed to scale: default is 1 tentacle (easy). Increase to up to max_tentacles (10).

*** MODIFIED FOR STAGE 0A: APPROACH ONLY ***
- Simplified observation space to track only 1 ring/tentacle.
- Gripper is locked open.
- Reward is highly scaled distance reduction (500.0 * delta_distance).
"""

import os
import sys
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import gymnasium as gym

# import robot class from repo
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ur5_with_gripper")) 


from robot import UR5Robotiq85

# ---------- write a torus OBJ (used to create a real ring/torus) ----------
def write_torus_obj(path, R=0.035, r=0.01, n_major=48, n_minor=20):
    """
    Write a torus mesh (.obj) to 'path'. R = major radius, r = minor radius.
    Resolution: n_major x n_minor 
    """
    verts = []
    faces = []
    for i in range(n_major):
        theta = (i / n_major) * 2 * math.pi
        for j in range(n_minor):
            phi = (j / n_minor) * 2 * math.pi
            x = (R + r * math.cos(phi)) * math.cos(theta)
            y = (R + r * math.cos(phi)) * math.sin(theta)
            z = r * math.sin(phi)
            verts.append((x, y, z))
    for i in range(n_major):
        for j in range(n_minor):
            i2 = (i + 1) % n_major
            j2 = (j + 1) % n_minor
            v0 = i * n_minor + j
            v1 = i2 * n_minor + j
            v2 = i2 * n_minor + j2
            v3 = i * n_minor + j2
            faces.append((v0+1, v1+1, v2+1))
            faces.append((v0+1, v2+1, v3+1))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for fa in faces:
            f.write(f"f {fa[0]} {fa[1]} {fa[2]}\n")

# ---------- class for creation of the ring pick and place environment ----------
class RingPickPlaceEnv(gym.Env):

    # ---------------- INITIALIZE ----------------
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
        self.gui = gui
        self.timestep = timestep
        self.frame_skip = frame_skip

        # scene layout
        self.num_tentacles = num_tentacles
        self.max_tentacles = max_tentacles
        self.base_circle_radius = base_circle_radius
        self.tentacle_segments = tentacle_segments
        self.tentacle_segment_length = tentacle_segment_length
        self.tentacle_curve = tentacle_curve

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

        self.workspace_low = np.array([0.2, -0.45, 0.0])
        self.workspace_high = np.array([0.8, 0.45, 0.5])

        # physics
        if self.gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.timestep)

        # reward bookkeeping
        self.holding = False
        self.hold_constraint = None
        self.held_ring_id = None
        self.max_delta = 0.02
        self.grasp_distance = 0.2
        self.place_distance = 0.2
        self.success_reward = 5.0
        self.spawn_zone_center = np.array([0.35, -0.35, 0.06])
        self.spawn_zone_size = np.array([0.1, 0.2, 0.02])

        # bookkeeping for bodies
        self.tentacle_ids = []          # list of body ids for current tentacle segments (flat list)
        self.tentacle_top_positions = []# list of 3-d positions (one per tentacle)
        self.tentacle_color_idxs = []   # list of color indices for each tentacle
        self.ring_ids = []              # list of ring body ids (one per tentacle)
        self.spawn_zone_center = np.array([0.35, -0.35, 0.06])  # where rings spawn (near gripper)
        self.spawn_zone_size = np.array([0.1, 0.2, 0.02])      # region dims for ring spawn

        # load mesh if needed
        self._ensure_torus_mesh()

        # load scene + robot (via repo robot class!)
        self._load_scene()

        # action space: dx, dy, dz, grip
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # observation dim: 7 + 7 + 3 + 4 + 3 + 3 + 1 = 28 (Simplified for 1 target)
        num_joints = len(self.robot.controllable_joints)
        self.num_joints = num_joints
        obs_dim = num_joints + num_joints + 3 + 4 + 7
        obs_high = np.ones(obs_dim, dtype=np.float32) * 10
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.seed()

    # ---------------- ROBOT & SCENE LOADING ----------------
    def _load_scene(self):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")

        # use repo robot class
        self.robot = UR5Robotiq85(pos=[0,0,0], ori=[0,0,0])
        self.robot.step_simulation = lambda: p.stepSimulation() # need to use this to make ur5 environment like the kuka

        self.robot.load()  # calls __init_robot__, parse joints, etc.

        #self.ee_link_index = self.robot.eef_id
        self.ee_link_index = 12 

        self.num_joints = len(self.robot.controllable_joints)

        # clear bodies
        self.tentacle_ids = []
        self.tentacle_top_positions = []
        self.tentacle_color_idxs = []
        self.ring_ids = []
        self.holding = False
        self.hold_constraint = None
        self.held_ring_id = None

    # ---------------- TENTACLE POSITION ARRANGEMENT ----------------
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

    # ---------------- CAMERA CONTROLS ----------------
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

    # ---------------- stepping / grasping ----------------
    def _attach_ring(self, ring_id):
        if self.hold_constraint is None and ring_id is not None:
            cid = p.createConstraint(
                parentBodyUniqueId=self.robot.id,  
                parentLinkIndex=self.ee_link_index,
                childBodyUniqueId=ring_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0,0,0],
                parentFramePosition=[0,0,0],
                childFramePosition=[0,0,0]
            )
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

    # ---------------- UTILITIES ----------------
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _ensure_torus_mesh(self):
        mesh_path = os.path.join(os.path.dirname(__file__), "torus.obj")
        if not os.path.exists(mesh_path):
            write_torus_obj(mesh_path)
        self._torus_path = mesh_path

    # ---------------- GET EE POSE OF UR5E ----------------
    def _get_ee_pose(self):
        state = p.getLinkState(self.robot.id, self.robot.eef_id, computeForwardKinematics=True)
        pos = np.array(state[4])
        orn = np.array(state[5])
        return pos, orn
    
    # ---------------- DELTA ACTION ROBOT CONTROL ----------------
    def _apply_action(self, action):
        action = np.clip(action, -1, 1)
        dx, dy, dz, grip_raw = action
        delta = np.array([dx, dy, dz]) * self.max_delta

        ee_pos, ee_orn = self._get_ee_pose()
        target_pos = ee_pos + delta
        target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)

        # fixed downward orientation so grasping is possible
        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        # call into repo robot API
        self.robot.move_ee([*target_pos, *p.getEulerFromQuaternion(target_orn)], control_method="end")

        # gripper control (FOR STAGE 0A, THIS IS OVERRIDDEN IN STEP)
        if grip_raw > 0:
            self.robot.close_gripper()
        else:
            self.robot.open_gripper()

        for _ in range(self.frame_skip):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.timestep)

        return grip_raw > 0  # logical closed/open
    
    # --- Helper function for checking placement (Must be defined in your class) ---
    def _get_nearest_ring_distance(self, ee_pos):
        # A simple helper function to keep the logic clean
        if not self.ring_ids:
            return 1e6, None
        
        # Assumes single ring target
        rid = self.ring_ids[0]
        try:
            rpos, _ = p.getBasePositionAndOrientation(rid)
            d = np.linalg.norm(np.array(rpos) - ee_pos)
            return d, rid
        except:
            return 1e6, None

    def _is_ring_placed(self):
        # Checks if the ring is on the target tentacle
        if not self.tentacle_top_positions or not self.ring_ids:
            return False
            
        tentacle_base_pos = np.array(self.tentacle_top_positions[0])
        ring_pos, _ = p.getBasePositionAndOrientation(self.ring_ids[0])
        ring_pos = np.array(ring_pos)

        # Check: Ring is low enough AND horizontally aligned with the tentacle
        # Z Check: Ring Z is close to the ground/base of the tentacle segments
        z_check = ring_pos[2] < tentacle_base_pos[2] 
        # XY Check: Ring center is within a small radius of the tentacle base (0.05m tolerance)
        xy_check = np.linalg.norm(ring_pos[:2] - tentacle_base_pos[:2]) < 0.05
        
        return xy_check and z_check

    # ---------------- STEP FUNCTION (MODIFIED FOR STAGE 0A) ----------------
    def step(self, action):
            """
            Stage 0C: Full Task.
            - Episode ends only on successful placement.
            - Reward switches based on whether the agent is 'holding' the ring.
            """
            if not hasattr(self, "step_count"): self.step_count = 0
            if not hasattr(self, "prev_nearest_d_ring"): self.prev_nearest_d_ring = None
            if not hasattr(self, "prev_nearest_d_tentacle"): self.prev_nearest_d_tentacle = None
            
            self.step_count += 1

            # Parse action and apply movement (same as Stage 0B)
            action = np.clip(action, -1.0, 1.0)
            dx, dy, dz, grip_action = action
            delta = np.array([dx, dy, dz], dtype=np.float32) * self.max_delta

            ee_pos, ee_orn = self._get_ee_pose()
            target_pos = ee_pos + delta
            target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)
            self.robot.move_ee([target_pos[0], target_pos[1], target_pos[2], 3.14, 0, 0],
                            control_method='end')

            # --- Physics Step ---
            for _ in range(self.frame_skip): p.stepSimulation()

            # Get new EE position and target positions
            ee_pos, ee_orn = self._get_ee_pose()
            tentacle_top_pos = self.tentacle_top_positions[0] # Assumes num_tentacles=1
            
            # --- Grasp Attempt/Control ---
            grasped_this_step = False
            ring_release_attempt = False
            
            if not self.holding:
                # 1. ATTEMPT GRASP (same as Stage 0B)
                nearest_d_ring = self._get_nearest_ring_distance(ee_pos)[0]
                if grip_action > 0.5 and nearest_d_ring < self.grasp_distance:
                    self._attach_ring(self.ring_ids[0]) # Assume ring_ids[0] is the target
                    grasped_this_step = True
                jaw_opening = 0.0 if grip_action > 0.5 else 0.085
            else:
                # 2. RELEASE MECHANISM (Triggered by negative grip action near target Z)
                nearest_d_ring = 1e6 # Ring distance is irrelevant when holding
                d_to_tentacle_xy = np.linalg.norm(ee_pos[:2] - tentacle_top_pos[:2])
                
                # Agent tries to open gripper when over the tentacle
                if grip_action < -0.5 and ee_pos[2] < tentacle_top_pos[2] + 0.03 and d_to_tentacle_xy < 0.1:
                    self._release_ring()
                    ring_release_attempt = True
                
                jaw_opening = 0.0 # Gripper is conceptually closed while holding
            
            self.robot.move_gripper(jaw_opening)
            
            # Update trackers
            if not self.holding:
                self.prev_nearest_d_ring = nearest_d_ring
            d_ee_to_tentacle = np.linalg.norm(ee_pos - tentacle_top_pos)
            if self.prev_nearest_d_tentacle is None: self.prev_nearest_d_tentacle = d_ee_to_tentacle

            # ============================================
            # MODIFIED REWARD FOR SUB-STAGE 0A (Approach Only)
            # ============================================
            reward = 0.0
            done = False
            info = {}

            if not self.holding:
                # 1. Inverse Distance Penalty (R_State)
                # Penalizes distance to provide a stable value gradient
                reward_scale = 30.0 # Tune this magnitude
                reward += -reward_scale * nearest_d_ring


                # 3. Success Bonus (R_Success)
                if nearest_d_ring < 0.2:
                    reward += 50.0

                grip_bonus_scale = 40.0 
                if nearest_d_ring < self.grasp_distance:
                    # We want to reward positive grip actions (closing) proportionally.
                    # Use max(0, grip_action) to ensure we don't reward opening.
                    reward += grip_bonus_scale * np.clip(grip_action, 0, 1)
                
                # 4. BONUS: HUGE bonus for successful grasp (R_Grasp)
                if grasped_this_step:
                    reward += 100.0
                    info['grasped'] = True

            if self.holding:
                # 1. Distance Reduction Reward (R_Hold_Distance)
                # Reward based on reduction in distance to tentacle top
                reward_scale = 50.0 # Tune this magnitude
                reward += -reward_scale * d_ee_to_tentacle

                if d_ee_to_tentacle < self.place_distance:
                    reward += 100.0

                # grip_bonus_scale = 40.0 
                # if d_ee_to_tentacle > self.place_distance:
                #     # We want to reward positive grip actions (closing) proportionally.
                #     # Use max(0, grip_action) to ensure we don't reward opening.
                #     reward += grip_bonus_scale * np.clip(grip_action, -1, 0)

                # if d_ee_to_tentacle < self.place_distance:
                #     # We want to reward positive grip actions (closing) proportionally.
                #     # Use max(0, grip_action) to ensure we don't reward opening.
                #     reward += -grip_bonus_scale * np.clip(grip_action, -1, 0)

                # # 2. Success Bonus for Placement (R_Place_Success)
                # if self._is_ring_placed():
                #     reward += 2000.0
                #     done = True
                #     info['success'] = True
                #     info['placed'] = True

            # Update tracker
            self.prev_nearest_d = nearest_d_ring
            
            # Build observation
            obs = self._get_obs()
            
            return obs.astype(np.float32), float(reward), done, False, info

    # ---------------- OBS CONSTRUCTION (Simplified) ----------------
    def _get_obs(self):
        # 1. Robot State
        joint_obs = self.robot.get_joint_obs()
        positions = np.array(joint_obs['positions'], dtype=np.float32)
        velocities = np.array(joint_obs['velocities'], dtype=np.float32)
        ee_pos = np.array(joint_obs['ee_pos'], dtype=np.float32)
        ee_orn = np.array(joint_obs['ee_orn'], dtype=np.float32) if 'ee_orn' in joint_obs else np.zeros(4, dtype=np.float32)

        # 2. Target State (Only the first/single target is relevant)
        
        # Tentacle Top Position
        if len(self.tentacle_top_positions) > 0:
            tpos = np.array(self.tentacle_top_positions[0], dtype=np.float32)
        else:
            tpos = np.zeros(3, dtype=np.float32)
            
        # Ring Position
        if len(self.ring_ids) > 0:
            try:
                rpos, _ = p.getBasePositionAndOrientation(self.ring_ids[0])
                rpos = np.array(rpos, dtype=np.float32)
            except Exception:
                rpos = np.zeros(3, dtype=np.float32)
        else:
            rpos = np.zeros(3, dtype=np.float32)
            
        # Color Index
        if len(self.tentacle_color_idxs) > 0:
            color_idx = float(self.tentacle_color_idxs[0])
        else:
            color_idx = -1.0
            
        # Target Obs Dimensions: 3 (tpos) + 3 (rpos) + 1 (color_idx) = 7
        target_obs = np.concatenate([tpos, rpos, [color_idx]], dtype=np.float32)

        # 3. Concatenate all observations
        # Dimensions: 7 + 7 + 3 + 4 + 7 = 28
        return np.concatenate([positions, velocities, ee_pos, ee_orn, target_obs]).astype(np.float32)
   
    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Reload robot + clear old tentacles/rings
        self._load_scene()

        # clear bookkeeping
        self.tentacle_ids = []
        self.tentacle_top_positions = []
        self.tentacle_color_idxs = []
        self.ring_ids = []
        self.holding = False
        self.hold_constraint = None
        self.held_ring_id = None

        # ensure robot is reset
        self.robot.reset()

        # determine number of tentacles this episode
        num_t = self.num_tentacles
        num_t = int(max(1, min(self.max_tentacles, num_t)))

        # arrange the tentacles in a circle
        poses = self._arrange_tentacle_positions(num_t)

        # choose color pattern with random start index
        palette = self.color_palette
        palette_len = len(palette)
        try:
            start_idx = int(self.np_random.integers(0, palette_len))
        except AttributeError:
            start_idx = int(self.np_random.randint(0, palette_len))

        color_idxs = [(start_idx + i) % palette_len for i in range(num_t)]
        self.tentacle_color_idxs = color_idxs

        # create tentacles
        for i, (x, y, z, yaw) in enumerate(poses):
            color = palette[color_idxs[i]]
            ids, top_pos = self._create_tentacle(
                [x, y, z],
                color_rgba=color,
                segments=self.tentacle_segments,
                seg_len=self.tentacle_segment_length,
                base_radius=0.045,
                tip_radius=0.012,
                curve=self.tentacle_curve,
                axis='y'
            )
            self.tentacle_ids.extend(ids)
            self.tentacle_top_positions.append(top_pos)

        # spawn rings near the robot spawn zone
        for cid in color_idxs:
            color = palette[cid]
            body, pos = self._spawn_ring_for_color(self.spawn_zone_center, color)
            self.ring_ids.append(body)

        # optional GUI camera
        if self.gui:
            center = np.array([(self.workspace_low[0] + self.workspace_high[0]) / 2.0, 0.0, 0.08])
            self._set_camera(distance=0.45, yaw=60, pitch=-30, target=center.tolist())

        # build initial observation
        obs = self._get_obs().astype(np.float32)

        return obs, {}

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    # Test with GUI to ensure basic function
    env = RingPickPlaceEnv(gui=True, num_tentacles=1)
    obs, info = env.reset()

    for _ in range(400):
        # Action only moves, grip is ignored in the step function
        a = env.action_space.sample() 
        obs, reward, done, truncated, info = env.step(a)

        if done or truncated:
            print(f"Episode done. Success: {info.get('success', False)}, Reward: {reward}")
            obs, info = env.reset()

    env.close()