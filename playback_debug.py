# playback_debug_fixed.py
import os, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv
import pybullet as p

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage1_final.zip")
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")

# Create GUI env wrapped by Monitor (same as training)
raw_env = DummyVecEnv([lambda: Monitor(RingPickPlaceEnv(gui=True, num_tentacles=1))])
env = VecNormalize.load(vec_path, raw_env)
env.training = False
env.norm_reward = False

# load model and attach env
model = PPO.load(model_path)
model.set_env(env)

obs = env.reset()

# Helper to fetch the underlying RingPickPlaceEnv instance:
# env is VecNormalize -> .venv is the underlying VecEnv (DummyVecEnv) -> .envs[0] is Monitor -> .env is the real env
vecenv = env.venv              # VecNormalize.venv -> DummyVecEnv
monitor0 = vecenv.envs[0]      # Monitor wrapper
base_env = monitor0.env        # the actual RingPickPlaceEnv instance

# debug params
PRINT_EVERY = 1
GRIP_SMOOTH = True       # toggle to test whether grip flicker is the culprit
SMOOTH_HOLD_STEPS = 6    # if smoothing, keep positive-grip for this many steps

# smoothing state
hold_counter = 0
smoothed_grip = 0.0

def get_nearest_ring_info(env_instance):
    try:
        ee_pos, _ = env_instance._get_ee_pose()
    except Exception:
        return None, float('inf')
    nearest = None
    nd = float('inf')
    for rid in list(env_instance.ring_ids):
        try:
            rpos, _ = p.getBasePositionAndOrientation(rid)
        except Exception:
            continue
        d = np.linalg.norm(np.array(rpos) - np.array(ee_pos))
        if d < nd:
            nd = d
            nearest = rid
    return nearest, nd

# Monkeypatch attach/release on the real env (base_env), not on the VecNormalize wrapper
orig_attach = base_env._attach_ring
orig_release = base_env._release_ring
def attach_and_log(rid):
    print("========> ATTACH called for rid", rid)
    return orig_attach(rid)
def release_and_log():
    print("========> RELEASE called")
    return orig_release()
base_env._attach_ring = attach_and_log
base_env._release_ring = release_and_log


# ------------------ REACHABILITY TEST (paste here) ------------------
import math

def try_reach_ring(base_env, rid, steps=200, motor_force=600):
    """Attempt to move the EE to the ring's base position using IK and strong motors.
    Returns final EE->ring distance (meters)."""
    if rid is None:
        print("No ring id given for reach test.")
        return float('inf')
    try:
        rpos, _ = p.getBasePositionAndOrientation(rid)
    except Exception as e:
        print("Could not get ring pos:", e)
        return float('inf')

    # Run a short controller loop to move to the ring position
    for _ in range(steps):
        # compute IK for ring position (use many iterations)
        target_joints = p.calculateInverseKinematics(base_env.kuka,
                                                     base_env.ee_link_index,
                                                     rpos,
                                                     maxNumIterations=200)
        # apply target positions with high force so it can try to reach quickly
        for j in range(len(target_joints)):
            p.setJointMotorControl2(bodyUniqueId=base_env.kuka,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_joints[j],
                                    force=motor_force)
        p.stepSimulation()

    # measure final distance
    try:
        ee_pos, _ = base_env._get_ee_pose()
    except Exception as e:
        print("Could not read EE pose after reach test:", e)
        return float('inf')
    d = math.dist(ee_pos, rpos)
    return d

# find nearest ring id from base_env (same helper as loop uses)
nearest_id, nearest_d = get_nearest_ring_info(base_env)
print("REACH TEST: nearest_id=", nearest_id, "initial distance=", nearest_d)

if nearest_id is not None:
    d_after = try_reach_ring(base_env, nearest_id, steps=300, motor_force=800)
    print("REACH TEST result: distance after forced IK attempt = {:.4f} m".format(d_after))
    if d_after < base_env.grasp_distance:
        print("REACH TEST: SUCCESS — EE CAN reach the ring (d_after < grasp_distance).")
    else:
        print("REACH TEST: FAILURE — EE CANNOT reach the ring (d_after >= grasp_distance).")
else:
    print("REACH TEST: no ring available to test.")
# ------------------ end reachability test block ------------------



try:
    for step_i in range(2000):
        action, _ = model.predict(obs, deterministic=True)

        # optional smoothing of the grip channel (action[-1] is grip)
        # handle batched action shape: (n_envs, act_dim)
        a = np.array(action, copy=True)
        if GRIP_SMOOTH:
            raw_grip = float(a[0, -1])
            if raw_grip > 0.0:
                smoothed_grip = 1.0
                hold_counter = SMOOTH_HOLD_STEPS
            else:
                if hold_counter > 0:
                    hold_counter -= 1
                    smoothed_grip = 1.0
                else:
                    smoothed_grip = raw_grip
            a[0, -1] = smoothed_grip
        action_to_step = a

        # diagnostics from the unwrapped env
        nearest_id, nearest_d = get_nearest_ring_info(base_env)
        holding_flag = base_env.holding
        held_id = base_env.held_ring_id

        an = a[0]
        if step_i % PRINT_EVERY == 0:
            print(f"step {step_i:04d} action_norm={np.linalg.norm(an):.4f} action={an} nearest_id={nearest_id} nearest_d={nearest_d:.3f} holding={holding_flag} held_id={held_id}")

        # step the VecNormalize-wrapped env
        step_result = env.step(action_to_step)
        # gymnasium style handling
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, infos = step_result
            if isinstance(terminated, (np.ndarray, list)):
                done = bool(terminated[0] or truncated[0])
            else:
                done = bool(terminated or truncated)
        else:
            obs, rewards, done, infos = step_result
            if isinstance(done, (np.ndarray, list)):
                done = bool(done[0])
            else:
                done = bool(done)

        if done:
            print("Episode done:", infos)
            obs = env.reset()

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    env.close()
