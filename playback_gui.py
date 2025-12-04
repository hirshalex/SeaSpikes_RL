# playback_gui.py  (fixed for VecNormalize)
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv

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

    obs = env.reset()

    try:
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            
            # VecNormalize returns 4 values (old gym API style)
            obs, reward, done, infos = env.step(action)
            
            # Extract from arrays (vectorized env)
            done = done[0]
            reward_val = float(reward[0])

            if step % 50 == 0:
                print(f"step {step} reward={reward_val:.2f} done={done}")

            if done:
                info = infos[0]
                success = info.get('success', False)
                print(f"Episode finished - Success: {success}")
                obs = env.reset()
                time.sleep(1)  # pause between episodes

    except KeyboardInterrupt:
        print("Playback interrupted by user")
    finally:
        env.close()
        print("Closed env")