# debug_playback.py - See what the agent is actually doing (IMPROVED)
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env_ur5 import RingPickPlaceEnv
import pybullet as p

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage0d_final.zip")
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage0d.pkl")

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
    print("DEBUG PLAYBACK - Full Agent Behavior Analysis")
    print("=" * 60)


    obs = env.reset()

    base_env = env.envs[0].env  # underlying PyBullet env

    episode = 0
    step = 0
    ep_reward = 0

    was_holding = False

    try:
        while episode < 3:
            action, _ = model.predict(obs, deterministic=True)
            raw_action = action[0]

            dx, dy, dz, grip = raw_action

            obs, reward, done, infos = env.step(action)

            done = done[0]
            reward_val = float(reward[0])
            ep_reward += reward_val

            # ===============================
            # Collect core environment info
            # ===============================
            ee_pos, _ = base_env._get_ee_pose()

            # Ring info
            ring_d = 1e6
            ring_pos = None
            if base_env.ring_ids:
                ring_pos, _ = p.getBasePositionAndOrientation(base_env.ring_ids[0])
                ring_pos = np.array(ring_pos)
                ring_d = np.linalg.norm(ring_pos - ee_pos)

            # Tentacle info
            tentacle_top = None
            tentacle_d = None
            if base_env.tentacle_top_positions:
                tentacle_top = np.array(base_env.tentacle_top_positions[0])
                tentacle_d = np.linalg.norm(ee_pos - tentacle_top)

            # Ring → Tentacle distance (useful when holding)
            ring_to_tentacle_d = None
            if (ring_pos is not None) and (tentacle_top is not None):
                ring_to_tentacle_d = np.linalg.norm(ring_pos - tentacle_top)

            holding = base_env.holding

            # ====================================================
            # Detect grasp or drop transitions (holding changes)
            # ====================================================
            if holding and not was_holding:
                print(f"\n{'='*60}")
                print(f"[GRASP] Step {step}: Ring successfully attached!")
                print(f"  Grip action: {grip:.3f}")
                print(f"  EE→Ring distance at grasp: {ring_d:.3f}")
                print(f"{'='*60}\n")

            if was_holding and not holding:
                print(f"\n{'='*60}")
                if grip < -0.5:
                    print(f"[RELEASE] Step {step}: Intentional release")
                else:
                    print(f"[DROP] Step {step}: ACCIDENTAL drop!")
                print(f"  Grip action: {grip:.3f}")
                print(f"  EE→Ring dist at drop: {ring_d:.3f}")
                print(f"{'='*60}\n")

            was_holding = holding

            # ====================================================
            # DETAILED DEBUG EVERY 20 STEPS
            # ====================================================
            if step % 20 == 0:
                print(f"\n--- Step {step} ---")
                print(f"  Action: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}, grip={grip:.3f}")
                print(f"  Action magnitude: {np.linalg.norm([dx, dy, dz]):.3f}")

                print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

                print(f"  EE → Ring distance: {ring_d:.3f}")
                if tentacle_d is not None:
                    print(f"  EE → Tentacle distance: {tentacle_d:.3f}")
                if ring_to_tentacle_d is not None:
                    print(f"  Ring → Tentacle distance: {ring_to_tentacle_d:.3f}")

                print(f"  Holding ring: {holding}")
                print(f"  Reward this step: {reward_val:.3f}")
                print(f"  Episode cumulative reward: {ep_reward:.2f}")

                # --- Diagnostics ---
                if not holding:
                    if ring_d > 0.3:
                        print("  ⚠️  Far from ring — approach failing.")
                    elif ring_d < base_env.grasp_distance:
                        print("  ✓ In grasp range — should close soon.")
                    if grip > 0.5:
                        print("    ✓ Attempting to CLOSE gripper")
                    elif abs(grip) < 0.3:
                        print("    ⚠️ Neutral grip — not trying to grasp")
                else:
                    # When holding:
                    if ring_to_tentacle_d is not None:
                        if ring_to_tentacle_d > 0.2:
                            print("  ⚠️  Far from tentacle — placement phase failing")
                        else:
                            print("  ✓ Close to tentacle — should OPEN soon")

                    if grip < -0.5:
                        print("    ✓ Attempting to OPEN gripper (placement)")
                    elif abs(grip) < 0.3:
                        print("    ⚠️ Neutral grip while holding — no intent to release yet")

            # ====================================================
            # Episode termination
            # ====================================================
            if done:
                info = infos[0]
                success = info.get('success', False)

                print(f"\n{'='*60}")
                print(f"EPISODE {episode+1} FINISHED")
                print(f"  Total steps: {step}")
                print(f"  Episode reward: {ep_reward:.2f}")
                print(f"  Success: {success}")
                print(f"{'='*60}\n")

                obs = env.reset()
                episode += 1
                step = 0
                ep_reward = 0
                time.sleep(1)

            else:
                step += 1

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")

    finally:
        env.close()
        print("Closed env")
