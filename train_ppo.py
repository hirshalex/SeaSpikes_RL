# train_headless.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from pybullet_ring_env import RingPickPlaceEnv

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(num_tentacles=1):
    def _init():
        return Monitor(RingPickPlaceEnv(gui=False, num_tentacles=num_tentacles))
    return _init

if __name__ == "__main__":
    # train on 1 tentacle stage (headless)
    num_tentacles = 1
    train_env = DummyVecEnv([make_env(num_tentacles=num_tentacles)])
    vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=1024, batch_size=64, n_epochs=8,
                policy_kwargs=dict(net_arch=[256,256]), tensorboard_log="./ppo_tb")

    # checkpoint callback (periodic)
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=MODEL_DIR, name_prefix=f"ppo_stage1")

    model.learn(total_timesteps=200_000, callback=checkpoint_cb)

    # Save final model and the VecNormalize wrapper
    model_path = os.path.join(MODEL_DIR, "ppo_stage1_final")
    model.save(model_path)
    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")
    vec_env.save(vecsave_path)

    vec_env.close()
