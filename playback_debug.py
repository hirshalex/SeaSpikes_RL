# debug_playback.py - See what the agent is actually doing
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv
import pybullet as p

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage1_final.zip")
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")

def make_gui_env():
    def _init():
        env = RingPickPlaceEnv(gui=True, num_tentacles=1)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    raw_env = DummyVecEnv([make_gui_env()])

    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, raw_env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: vecnormalize file not found at", vec_path)
        env = raw_env

    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=" * 60)
    print("DEBUG PLAYBACK - Detailed Agent Behavior Analysis")
    print("=" * 60)
    print("Watch for:")
    print("  - Is the gripper action ever positive (trying to close)?")
    print("  - Does distance to ring decrease over time?")
    print("  - Does the agent ever get close enough to grasp?")
    print("=" * 60)

    obs = env.reset()

    # Get access to underlying env for debug info
    base_env = env.envs[0].env

    episode = 0
    step = 0
    ep_reward = 0

    try:
        while episode < 3:  # Run 3 episodes
            action, _ = model.predict(obs, deterministic=True)
            
            # EXTRACT RAW ACTION VALUES
            raw_action = action[0]  # Get first env's action
            dx, dy, dz, grip = raw_action
            
            obs, reward, done, infos = env.step(action)
            
            done = done[0]
            reward_val = float(reward[0])
            ep_reward += reward_val

            # GET DEBUG INFO FROM ENVIRONMENT
            ee_pos, _ = base_env._get_ee_pose()
            
            # Find nearest ring
            nearest_d = 1e6
            if base_env.ring_ids:
                ring_pos, _ = p.getBasePositionAndOrientation(base_env.ring_ids[0])
                nearest_d = np.linalg.norm(np.array(ring_pos) - ee_pos)
            
            holding = base_env.holding
            
            # Print detailed info every 20 steps
            if step % 20 == 0:
                print(f"\n--- Step {step} ---")
                print(f"  Action: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, grip={grip:.3f}")
                print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                print(f"  Distance to ring: {nearest_d:.3f}")
                print(f"  Holding: {holding}")
                print(f"  Reward: {reward_val:.3f}")
                print(f"  Cumulative reward: {ep_reward:.2f}")
                
                # DIAGNOSE ISSUES
                if abs(grip) < 0.3:
                    print(f"  ⚠️  Gripper action is neutral (not trying to grasp or release)")
                if grip > 0.5:
                    print(f"  ✓ Trying to CLOSE gripper")
                    if nearest_d > base_env.grasp_distance:
                        print(f"    ❌ But too far from ring (need < {base_env.grasp_distance:.3f})")
                if grip < -0.5:
                    print(f"  ✓ Trying to OPEN gripper")
                
                if nearest_d > 0.3:
                    print(f"  ⚠️  Agent is far from ring - approach phase failing")
                elif nearest_d < base_env.grasp_distance and not holding:
                    print(f"  ✓ In grasp range! Should try to close gripper")

            if done:
                info = infos[0]
                success = info.get('success', False)
                print(f"\n{'='*60}")
                print(f"EPISODE {episode + 1} FINISHED")
                print(f"  Total steps: {step}")
                print(f"  Total reward: {ep_reward:.2f}")
                print(f"  Success: {success}")
                print(f"{'='*60}\n")
                
                obs = env.reset()
                episode += 1
                step = 0
                ep_reward = 0
                time.sleep(2)
            else:
                step += 1

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    finally:
        env.close()
        print("Closed env")