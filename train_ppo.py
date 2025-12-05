# train_ppo.py - Improved training with better hyperparameters and logging
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


class RewardLoggingCallback(BaseCallback):
    """
    Custom callback to log reward components and success rate.
    Helps debug what the agent is learning.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        for i, done in enumerate(self.locals['dones']):
            if done:
                # Get info from the environment
                info = self.locals['infos'][i]
                
                # Log episode metrics
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Track success
                    success = 1.0 if info.get('success', False) else 0.0
                    self.episode_successes.append(success)
                    
                    # Log to tensorboard every episode
                    self.logger.record('rollout/ep_reward', ep_reward)
                    self.logger.record('rollout/ep_length', ep_length)
                    
                    # Log success rate (last 100 episodes)
                    if len(self.episode_successes) >= 100:
                        recent_success_rate = np.mean(self.episode_successes[-100:])
                        self.logger.record('rollout/success_rate', recent_success_rate)
                    
                    # Log grasp detection if available
                    if info.get('grasped', False):
                        self.logger.record('rollout/grasps', 1.0)
        
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

    # Parallel environments (increase for faster data collection)
    # More envs = more diverse experience per update
    n_envs = 8  # Increased from 4 (if your CPU can handle it)

    # Create vectorized environment with different seeds
    env_fns = [make_env_fn(num_tentacles=num_tentacles, seed=1000 + i)
               for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)

    # Normalization: normalize observations only (not rewards)
    # This helps with learning stability
    vec_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=False,  # Keep rewards unnormalized for interpretability
        clip_obs=10.0
    )

    # Monitor wrapper for logging
    vec_env = VecMonitor(vec_env)

    # ============================================
    # PPO HYPERPARAMETERS (tuned for manipulation)
    # ============================================
    policy_kwargs = dict(
        net_arch=[256, 256],  # Two hidden layers with 256 units each
        # Can also try dict(pi=[256, 256], vf=[256, 256]) for separate networks
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        
        # Rollout settings
        n_steps=2048,           # Keep this - good balance
        batch_size=128,         # Increased from 64 (larger batches = more stable)
        n_epochs=10,            # Keep this
        
        # Learning rates
        learning_rate=3e-4,     # Standard for PPO
        
        # PPO-specific
        clip_range=0.2,         # Keep this
        gae_lambda=0.95,        # Keep this
        gamma=0.99,             # Keep this
        
        # Exploration - START HIGHER, will naturally decay
        ent_coef=0.05,          # Increased from 0.01 for more initial exploration
                                # With better rewards, agent needs more exploration to discover grasp
        
        # Value function coefficient (helps with learning value estimates)
        vf_coef=0.5,            # Default but explicit
        max_grad_norm=0.5,      # Gradient clipping (default but explicit)
        
        # Network architecture
        policy_kwargs=policy_kwargs,
        
        # Logging
        tensorboard_log="./ppo_tb"
    )

    print("=" * 60)
    print("PPO Training Configuration:")
    print(f"  Environments: {n_envs}")
    print(f"  Tentacles: {num_tentacles}")
    print(f"  Total timesteps: 3,000,000")
    print(f"  Steps per rollout: {2048 * n_envs} ({2048} per env × {n_envs})")
    print(f"  Batch size: 128")
    print(f"  Updates per rollout: 10")
    print(f"  Entropy coefficient: 0.05 (high exploration)")
    print(f"  Learning rate: 3e-4")
    print("=" * 60)
    print("\nMonitor training progress:")
    print("  tensorboard --logdir ./ppo_tb")
    print("\nExpected timeline with improved rewards:")
    print("  ~50k steps: Agent starts approaching ring")
    print("  ~300k steps: First successful grasps")
    print("  ~800k steps: Consistent grasping")
    print("  ~1.5M steps: Full task completion")
    print("=" * 60)

    # ============================================
    # CALLBACKS
    # ============================================
    
    # Checkpoint callback - save every 250k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=250_000 // n_envs,  # Divide by n_envs because it counts steps per env
        save_path=MODEL_DIR,
        name_prefix="ppo_stage1_ckpt",
        verbose=1
    )
    
    # Custom reward logging callback
    reward_cb = RewardLoggingCallback()
    
    # Combine callbacks
    callback = CallbackList([checkpoint_cb, reward_cb])

    # ============================================
    # TRAIN
    # ============================================
    total_timesteps = 3_000_000
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback
            # progress_bar=True  # Requires tqdm and rich: pip install tqdm rich
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")

    # ============================================
    # SAVE FINAL MODEL
    # ============================================
    model_path = os.path.join(MODEL_DIR, "ppo_grap_v1")
    model.save(model_path)
    print(f"\nFinal model saved to: {model_path}")

    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage1.pkl")
    vec_env.save(vecsave_path)
    print(f"VecNormalize stats saved to: {vecsave_path}")

    vec_env.close()
    print("\nTraining complete!")
    print("\nTo evaluate your trained model, run:")
    print("  python playback_gui.py")
    print("or")
    print("  python debug_playback.py")