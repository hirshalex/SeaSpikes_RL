# train_ppo_stage0a.py - Training for Approach Only (Stage 0A) - FORCED FRESH START
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
# Import your environment
from pybullet_ring_env_ur5 import RingPickPlaceEnv 

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
                    # Log to tensorboard every episode
                    self.logger.record('rollout/ep_reward', info['episode']['r'])
                    self.logger.record('rollout/ep_length', info['episode']['l'])
                    
                    # Track success
                    success = 1.0 if info.get('success', False) else 0.0
                    self.episode_successes.append(success)
                    
                    # Log average success rate
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
    n_envs = 8 # Number of parallel environments

    # Create vectorized environment with different seeds
    env_fns = [make_env_fn(num_tentacles=num_tentacles, seed=1000 + i)
               for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)

    # Normalization: Essential for stable training. **START FRESH.**
    # Note: We are NOT loading an old VecNormalize file.
    vec_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=False, 
        clip_obs=10.0
    )

    vec_env = VecMonitor(vec_env)

    # ============================================
    # FORCING FRESH MODEL START (Stage 0A)
    # The 'if os.path.exists()' logic is removed.
    # ============================================
    print("Starting **FRESH** model for Stage 0A: Approach Only.")
    
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        
        # Hyperparameters (keep these)
        n_steps=2048,           
        batch_size=128,         
        n_epochs=10,            
        learning_rate=3e-4,     
        clip_range=0.2,         
        gae_lambda=0.95,        
        gamma=0.99,             
        ent_coef=0.1,           
        vf_coef=0.5,            
        max_grad_norm=0.5,      
        policy_kwargs=policy_kwargs,
        
        tensorboard_log="./ppo_tb"
    )
    
    print("\n" + "=" * 60)
    print("PPO TRAINING: SUB-STAGE 0A (APPROACH ONLY)")
    print("Forced start from scratch to fix max-action policy.")
    print("Goal: Achieve 90%+ success rate (getting within 5cm of ring).")
    print("=" * 60)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs, 
        save_path=MODEL_DIR,
        name_prefix="ppo_stage0a_ckpt",
        verbose=1
    )
    reward_cb = RewardLoggingCallback()
    callback = CallbackList([checkpoint_cb, reward_cb])

    # TRAIN
    total_timesteps = 500_000 
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback,
            reset_num_timesteps=True # Always reset the internal timestep counter
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")

    # SAVE FINAL MODEL
    model_path_stage0a = os.path.join(MODEL_DIR, "ppo_stage0a_final.zip")
    model.save(model_path_stage0a)
    print(f"\nFinal Stage 0A model saved to: {model_path_stage0a}")

    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage0a.pkl")
    vec_env.save(vecsave_path)
    print(f"VecNormalize stats saved to: {vecsave_path}")

    vec_env.close()
    print("\nTraining complete!")