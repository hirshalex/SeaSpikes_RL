# train_ppo_stage0b.py - Training for Approach + Grasp (Stage 0B)
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

# Define file paths based on the completed Stage 0A
MODEL_PATH_0A = os.path.join(MODEL_DIR, "ppo_stage0a_final.zip")
VECNORM_PATH_0A = os.path.join(MODEL_DIR, "vecnormalize_stage0a.pkl")
MODEL_PATH_0B = os.path.join(MODEL_DIR, "ppo_stage0b_final.zip")

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
    n_envs = 8 # Number of parallel environments

    # Create raw vectorized environment
    env_fns = [make_env_fn(num_tentacles=num_tentacles, seed=1000 + i)
               for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)

    # ============================================
    # LOAD VECNORM STATS FROM STAGE 0A
    # ============================================
    if not os.path.exists(VECNORM_PATH_0A):
        raise FileNotFoundError(f"VecNormalize stats not found at {VECNORM_PATH_0A}. Run Stage 0A training first!")

    vec_env = VecNormalize.load(VECNORM_PATH_0A, train_env)
    vec_env.norm_reward = False # Keep rewards unnormalized for this stage

    vec_env = VecMonitor(vec_env)

    # ============================================
    # LOAD POLICY FROM STAGE 0A CHECKPOINT
    # ============================================
    if not os.path.exists(MODEL_PATH_0A):
        raise FileNotFoundError(f"Stage 0A model not found at {MODEL_PATH_0A}. Cannot bootstrap Stage 0B.")

    print(f"Loading Stage 0A policy from: {MODEL_PATH_0A}")
    model = PPO.load(MODEL_PATH_0A, env=vec_env, tensorboard_log="./ppo_tb")

    # Ensure the model continues training from the loaded state
    model.n_steps = 2048 # Reset to desired n_steps if PPO object changed it

    model.ent_coef = 0.05 

    print("\n" + "=" * 60)
    print("PPO TRAINING: SUB-STAGE 0B (APPROACH + GRASP)")
    print("BOOTSTRAPPING policy from Stage 0A expertise.")
    print("Agent only needs to learn WHEN to close the gripper.")
    print("=" * 60)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs, # Less frequent checkpoints, we only need to learn the final move
        save_path=MODEL_DIR,
        name_prefix="ppo_stage0b_ckpt",
        verbose=1
    )
    reward_cb = RewardLoggingCallback()
    callback = CallbackList([checkpoint_cb, reward_cb])

    # TRAIN
    # Start with 200k steps, the agent should converge faster since it knows how to approach
    total_timesteps = 500_000 
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback,
            reset_num_timesteps=False # IMPORTANT: Continues step count from loaded model
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving current model...")

    # SAVE FINAL MODEL
    model.save(MODEL_PATH_0B)
    print(f"\nFinal Stage 0B model saved to: {MODEL_PATH_0B}")

    # Save VecNormalize stats (optional, but good practice)
    vecsave_path = os.path.join(MODEL_DIR, "vecnormalize_stage0b.pkl")
    vec_env.save(vecsave_path)
    print(f"VecNormalize stats saved to: {vecsave_path}")

    vec_env.close()
    print("\nStage 0B Training complete! Next step: Stage 1 (Lift & Move)")