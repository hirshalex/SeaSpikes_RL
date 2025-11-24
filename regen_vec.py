# regen_vec.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage1_final")  # no .zip in SB3 load is OK
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")

def make_env():
    return DummyVecEnv([lambda: Monitor(RingPickPlaceEnv(gui=False, num_tentacles=1))])

if __name__ == "__main__":
    # Build fresh VecNormalize (same API as current runtime)
    raw_env = make_env()
    vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=False)

    # Load model (if model has .zip suffix, remove or keep; SB3 can handle without .zip)
    model = PPO.load(model_path)

    # Run the model for a bit to accumulate observation statistics
    obs = vec_env.reset()
    steps = 2000  # enough to collect decent obs stats; increase if you like
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        out = vec_env.step(action)
        # vec_env.step returns vectorized outputs — unpack robustly:
        if len(out) == 5:
            obs, rewards, terminated, truncated, infos = out
            done = bool(np.array(terminated).any() or np.array(truncated).any())
        else:
            obs, rewards, done_raw, infos = out
            done = bool(np.array(done_raw).any())
        if done:
            obs = vec_env.reset()

    # Save vec stats that match your current environment implementation
    vec_env.save(vec_path)
    vec_env.close()
    print("Saved new VecNormalize to", vec_path)
