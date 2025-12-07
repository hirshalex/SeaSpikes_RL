import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

# Import your environment
from pybullet_ring_env_ur5 import RingPickPlaceEnv 

# ==========================================
# 🛑 CONFIGURATION SECTION (EDIT HERE) 🛑
# ==========================================

CONFIG = {
    # --- DIRECTORIES ---
    "MODEL_DIR": "models_dqn",
    "LOG_DIR":   "logs_dqn_overnight",

    # --- TRAINING DURATION (Timesteps) ---
    # FOR TEST: Use 5,000 | FOR OVERNIGHT: Use 500,000 (0A/0B) and 800,000 (0C)
    "STEPS_0A": 5_000_000,   
    "STEPS_0B": 5_000_000,
    "STEPS_0C": 8_000_000,

    # --- PARALLELISM ---
    # 8 is standard for Ryzen/Intel i7/i9. Lower to 1 or 2 if testing on a laptop.
    "N_ENVS": 8,

    # --- HYPERPARAMETERS ---
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 32,
    "GAMMA": 0.99,
    
    # IMPORTANT: 'learning_starts' must be SMALLER than 'STEPS' or no training happens!
    # FOR TEST: Use 1,000 | FOR OVERNIGHT: Use 10,000
    "LEARNING_STARTS": 10_000, 
    
    # Buffer Size (Replay Memory)
    # FOR TEST: Use 5,000 | FOR OVERNIGHT: Use 100,000
    "BUFFER_SIZE": 10_000,
}

# ==========================================
# 1. DISCRETE ACTION WRAPPER (Robust)
# ==========================================
class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Maps 8 discrete integers to continuous robot movements.
    Includes fixes for seeding and safe closing.
    """
    def __init__(self, env):
        super().__init__(env)
        # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z, 6: Open, 7: Close
        self.action_space = spaces.Discrete(8)
    
    def action(self, action):
        # Default: No movement (0,0,0), Open gripper (-1.0)
        act = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        
        if   action == 0: act[0] =  1.0  # +X
        elif action == 1: act[0] = -1.0  # -X
        elif action == 2: act[1] =  1.0  # +Y
        elif action == 3: act[1] = -1.0  # -Y
        elif action == 4: act[2] =  1.0  # +Z
        elif action == 5: act[2] = -1.0  # -Z
        elif action == 6: act[3] = -1.0  # Force Open
        elif action == 7: act[3] =  1.0  # Force Close
        return act
    
    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return []

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

# ==========================================
# 2. LOGGING CALLBACK
# ==========================================
class RewardLoggingCallback(BaseCallback):
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
                    recent_success = np.mean(self.episode_successes[-100:])
                    self.logger.record('rollout/success_rate', recent_success)
        return True

# ==========================================
# 3. UTILS & SETUP
# ==========================================
os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

# Define Paths Dynamically
PATH_0A_MODEL = os.path.join(CONFIG["MODEL_DIR"], "dqn_stage0a_final.zip")
PATH_0A_VEC   = os.path.join(CONFIG["MODEL_DIR"], "vecnormalize_stage0a.pkl")
PATH_0B_MODEL = os.path.join(CONFIG["MODEL_DIR"], "dqn_stage0b_final.zip")
PATH_0B_VEC   = os.path.join(CONFIG["MODEL_DIR"], "vecnormalize_stage0b.pkl")
PATH_0C_MODEL = os.path.join(CONFIG["MODEL_DIR"], "dqn_stage0c_final.zip")
PATH_0C_VEC   = os.path.join(CONFIG["MODEL_DIR"], "vecnormalize_stage0c.pkl")

def make_env_fn(num_tentacles=1, seed=None):
    def _init():
        env = RingPickPlaceEnv(gui=False, num_tentacles=num_tentacles)
        env = DiscreteActionWrapper(env)
        if seed is not None:
            env.seed(seed)
        return Monitor(env)
    return _init

def get_vec_env():
    env_fns = [make_env_fn(num_tentacles=1, seed=1000 + i) for i in range(CONFIG["N_ENVS"])]
    return DummyVecEnv(env_fns)

# ==========================================
# 4. TRAINING STAGES
# ==========================================

def train_stage_0a():
    print("\n" + "="*50)
    print(f"STAGE 0A: APPROACH ONLY ({CONFIG['STEPS_0A']} steps)")
    print("="*50)
    
    train_env = get_vec_env()
    vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env = VecMonitor(vec_env)

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=CONFIG["LEARNING_RATE"],
        buffer_size=CONFIG["BUFFER_SIZE"],
        learning_starts=CONFIG["LEARNING_STARTS"],
        batch_size=CONFIG["BATCH_SIZE"],
        gamma=CONFIG["GAMMA"],
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="./dqn_overnight_tb"
    )

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1000, CONFIG['STEPS_0A']//5), save_path=CONFIG["MODEL_DIR"], name_prefix="dqn_0a"),
        RewardLoggingCallback()
    ])
    
    model.learn(total_timesteps=CONFIG["STEPS_0A"], callback=callbacks)

    model.save(PATH_0A_MODEL)
    vec_env.save(PATH_0A_VEC)
    print(f"[Stage 0A] Saved to {PATH_0A_MODEL}")
    vec_env.close()


def train_stage_0b():
    print("\n" + "="*50)
    print(f"STAGE 0B: GRASP ({CONFIG['STEPS_0B']} steps)")
    print("="*50)

    if not os.path.exists(PATH_0A_VEC): raise FileNotFoundError("Stage 0A stats missing!")
    train_env = get_vec_env()
    vec_env = VecNormalize.load(PATH_0A_VEC, train_env)
    vec_env.norm_reward = False
    vec_env = VecMonitor(vec_env)

    if not os.path.exists(PATH_0A_MODEL): raise FileNotFoundError("Stage 0A model missing!")
    model = DQN.load(PATH_0A_MODEL, env=vec_env, tensorboard_log="./dqn_overnight_tb")

    # Refinement Params
    model.exploration_initial_eps = 0.3
    model.exploration_final_eps = 0.05
    model.exploration_fraction = 0.2

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1000, CONFIG['STEPS_0B']//5), save_path=CONFIG["MODEL_DIR"], name_prefix="dqn_0b"),
        RewardLoggingCallback()
    ])

    model.learn(total_timesteps=CONFIG["STEPS_0B"], callback=callbacks, reset_num_timesteps=False)

    model.save(PATH_0B_MODEL)
    vec_env.save(PATH_0B_VEC)
    print(f"[Stage 0B] Saved to {PATH_0B_MODEL}")
    vec_env.close()


def train_stage_0c():
    print("\n" + "="*50)
    print(f"STAGE 0C: PLACE ({CONFIG['STEPS_0C']} steps)")
    print("="*50)

    if not os.path.exists(PATH_0B_VEC): raise FileNotFoundError("Stage 0B stats missing!")
    train_env = get_vec_env()
    vec_env = VecNormalize.load(PATH_0B_VEC, train_env)
    vec_env.norm_reward = False
    vec_env = VecMonitor(vec_env)

    if not os.path.exists(PATH_0B_MODEL): raise FileNotFoundError("Stage 0B model missing!")
    model = DQN.load(PATH_0B_MODEL, env=vec_env, tensorboard_log="./dqn_overnight_tb")

    # Precision Params
    model.exploration_initial_eps = 0.1
    model.exploration_final_eps = 0.02
    model.exploration_fraction = 0.2

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1000, CONFIG['STEPS_0C']//5), save_path=CONFIG["MODEL_DIR"], name_prefix="dqn_0c"),
        RewardLoggingCallback()
    ])

    model.learn(total_timesteps=CONFIG["STEPS_0C"], callback=callbacks, reset_num_timesteps=False)

    model.save(PATH_0C_MODEL)
    vec_env.save(PATH_0C_VEC)
    print(f"[Stage 0C] Saved to {PATH_0C_MODEL}")
    vec_env.close()


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    print("Starting Training Pipeline with CONFIG:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    try:
        train_stage_0a()
        train_stage_0b()
        train_stage_0c()
        
        duration = time.time() - start_time
        print(f"\nALL STAGES COMPLETED in {duration/60:.2f} minutes.")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        raise e