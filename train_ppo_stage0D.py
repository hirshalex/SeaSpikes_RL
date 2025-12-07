# train_ppo_stage0c.py - Training for Pick, Lift, and Place (Stage 0C)
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from pybullet_ring_env_ur5 import RingPickPlaceEnv 

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define file paths based on the completed Stage 0B
MODEL_PATH_0B = os.path.join(MODEL_DIR, "ppo_stage0c_final.zip")
VECNORM_PATH_0B = os.path.join(MODEL_DIR, "vecnormalize_stage0c.pkl")
MODEL_PATH_0C = os.path.join(MODEL_DIR, "ppo_stage0d_final.zip")


class RewardLoggingCallback(BaseCallback):
    """ Custom callback to log success rate. """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.logger.record('rollout/ep_reward', info['episode']['r'])
                    self.logger.record('rollout/ep_length', info['episode']['l'])
                    
                    success = 1.0 if info.get('success', False) else 0.0
                    self.episode_successes.append(success)
                    
                    recent_success_rate = np.mean(self.episode_successes[-100:])
                    self.logger.record('rollout/success_rate', recent_success_rate)
        
        return True

def make_env_fn(num_tentacles=1, seed=None):
    """Create a single environment with proper seeding."""
    def _init():
        env = RingPickPlaceEnv(gui=False, num_tentacles=num_tentacles)
        if seed is not None:
            env.seed(seed)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    num_tentacles = 1
    n_envs = 12 # Number of parallel environments

    # Create raw vectorized environment
    env_fns = [make_env_fn(num_tentacles=num_tentacles, seed=1000 + i)
               for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)

    # ============================================
    # LOAD VECNORM STATS FROM STAGE 0B
    # ============================================
    if not os.path.exists(VECNORM_PATH_0B):
        raise FileNotFoundError(f"VecNormalize stats not found at {VECNORM_PATH_0B}. Run Stage 0C training first!")

    vec_env = VecNormalize.load(VECNORM_PATH_0B, train_env)
    vec_env.norm_reward = False 

    vec_env = VecMonitor(vec_env)

    # ============================================
    # LOAD POLICY FROM STAGE 0B CHECKPOINT
    # ============================================
    if not os.path.exists(MODEL_PATH_0B):
        raise FileNotFoundError(f"Stage 0C model not found at {MODEL_PATH_0B}. Cannot bootstrap Stage 0D.")

    print(f"Loading Stage 0B policy from: {MODEL_PATH_0B}")
    model = PPO.load(MODEL_PATH_0B, env=vec_env, tensorboard_log="./ppo_tb")

    # Set Entropy and n_steps
    model.n_steps = 2048
    new_ent_coef = 0.03 # FURTHER DECREASE ENTROPY for high-precision placement
    model.ent_coef = new_ent_coef
    print(f"Set model entropy coefficient (ent_coef) to: {new_ent_coef}")

    print("\n" + "=" * 60)
    print("PPO TRAINING: STAGE 0D (FULL PICK & PLACE)")
    print("Agent is trained to LIFT, MOVE, and PLACE.")
    print("Targeting 500k steps for full task convergence.")
    print("=" * 60)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs, 
        save_path=MODEL_DIR,
        name_prefix="ppo_stage0d_ckpt",
        verbose=1
    )
    reward_cb = RewardLoggingCallback()
    callback = CallbackList([checkpoint_cb, reward_cb])

    # TRAIN
    # 500k steps should be enough to converge from the B-Stage expertise
    total_timesteps = 3_000_000 
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback,
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")

    # SAVE FINAL MODEL
    model.save(MODEL_PATH_0C)
    print(f"\nFinal Stage 0D model saved to: {MODEL_PATH_0C}")

    # Save VecNormalize stats
    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage0d.pkl")
    vec_env.save(vecsave_path)
    print(f"VecNormalize stats saved to: {vecsave_path}")

    vec_env.close()
    print("\nStage 0C Training complete! You have achieved the single-ring task.")