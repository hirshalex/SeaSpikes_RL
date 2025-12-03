# train_ppo.py  (NO EVAL)
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from pybullet_ring_env import RingPickPlaceEnv

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env_fn(num_tentacles=1, seed=None):
    def _init():
        env = RingPickPlaceEnv(gui=False, num_tentacles=num_tentacles)
        if seed is not None:
            env.seed(seed)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    num_tentacles = 1

    # number of parallel environments (2–8 is fine for CPU)
    n_envs = 4

    env_fns = [make_env_fn(num_tentacles=num_tentacles, seed=1000 + i)
               for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)

    # normalization (observations only)
    vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # monitor -> logs episode rewards/lengths
    vec_env = VecMonitor(vec_env)
    #vec_env.envs[0].env.debug = True


    # PPO model
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=1e-3,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_tb"
    )

    # checkpoint callback (NO eval)
    checkpoint_cb = CheckpointCallback(
        save_freq=250_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_stage1_ckpt"
    )

    # Train: long run recommended
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    # Save final model & VecNormalize stats
    model_path = os.path.join(MODEL_DIR, "ppo_stage1_final")
    model.save(model_path)

    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")
    vec_env.save(vecsave_path)

    vec_env.close()
