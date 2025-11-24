# playback_gui.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from pybullet_ring_env import RingPickPlaceEnv
import numpy as np

MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "ppo_stage1_final.zip")   # or checkpoint
vec_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")

# create raw GUI env (unwrapped)
raw_env = DummyVecEnv([lambda: Monitor(RingPickPlaceEnv(gui=True, num_tentacles=1))])

# load VecNormalize stats into raw env so observations are scaled same as training
env = VecNormalize.load(vec_path, raw_env)
env.training = False   # important: disable further normalization updates
env.norm_reward = False

# # load the model and attach to the normalized env
model = PPO.load(model_path, env=env)

obs = env.reset()




for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    step_result = env.step(action)
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
        # Print info for the first (and usually only) env
        info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
        print("Episode done:", info0)
        obs = env.reset()

env.close()
