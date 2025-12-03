# playback_gui.py  (fixed)
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage1_final.zip")   # your saved model
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")  # saved VecNormalize

def make_gui_env():
    # single GUI env wrapped with Monitor (for episode info)
    def _init():
        env = RingPickPlaceEnv(gui=True, num_tentacles=1)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    # make a single raw GUI env (not normalized yet)
    raw_env = DummyVecEnv([make_gui_env()])

    # load VecNormalize wrapper and its saved stats so observations are normalized exactly like training
    # NOTE: VecNormalize.load expects a VecEnv (here raw_env)
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, raw_env)
        # ensure we do NOT update running stats during evaluation/playback
        env.training = False
        env.norm_reward = False
    else:
        # fallback: just use raw env (may behave differently if you trained with normalization)
        print("WARNING: vecnormalize file not found at", vec_path)
        env = raw_env

    # Load the model and attach the normalized env at load time so SB3 doesn't complain about num_envs
    # Important: use PPO.load(..., env=env) rather than load then set_env
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    # reset and run
    obs = env.reset()

    try:
        for step in range(2000):
            # model.predict expects the same observation shape as training
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)

            # gymnasium-style returns: (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, infos = step_result
                done = bool(terminated or truncated)
            else:
                # old gym (unlikely) fallback
                obs, reward, done, infos = step_result
                done = bool(done)

            # optional debug print
            if step % 50 == 0:
                print(f"step {step} reward={np.array(reward)} done={done}")

            if done:
                info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
                print("Episode finished:", info0)
                obs = env.reset()

    except KeyboardInterrupt:
        print("Playback interrupted by user")
    finally:
        env.close()
        print("Closed env")
