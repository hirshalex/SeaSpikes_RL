import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p  # Import pybullet to catch the specific error

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from pybullet_ring_env_ur5 import RingPickPlaceEnv 

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_DIR = "models_dqn"
STAGE = "0c"  # Change to "0a" or "0b" to see earlier stages

MODEL_PATH = os.path.join(MODEL_DIR, f"dqn_stage{STAGE}_final.zip")
VEC_PATH   = os.path.join(MODEL_DIR, f"vecnormalize_stage{STAGE}.pkl")

# ==========================================
# DISCRETE WRAPPER (With Fixes)
# ==========================================
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z, 6: Open, 7: Close
        self.action_space = spaces.Discrete(8)
    
    def action(self, action):
        act = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        if   action == 0: act[0] =  1.0
        elif action == 1: act[0] = -1.0
        elif action == 2: act[1] =  1.0
        elif action == 3: act[1] = -1.0
        elif action == 4: act[2] =  1.0
        elif action == 5: act[2] = -1.0
        elif action == 6: act[3] = -1.0
        elif action == 7: act[3] =  1.0
        return act

    # Pass seed through (Robustness Fix 1)
    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return []

    # Handle closing safely (Robustness Fix 2)
    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

# ==========================================
# ENV CREATION
# ==========================================
def make_gui_env():
    def _init():
        # gui=True to visualize
        env = RingPickPlaceEnv(gui=True, num_tentacles=1)
        env = DiscreteActionWrapper(env) 
        return Monitor(env)
    return _init

# ==========================================
# MAIN PLAYBACK LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Loading Stage {STAGE} DQN model...")

    env = DummyVecEnv([make_gui_env()])

    # Load Stats
    if os.path.exists(VEC_PATH):
        print(f"Loading normalization stats from: {VEC_PATH}")
        env = VecNormalize.load(VEC_PATH, env)
        env.training = False     
        env.norm_reward = False  
    else:
        print(f"WARNING: Stats not found at {VEC_PATH}. Behavior will be erratic.")

    # Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = DQN.load(MODEL_PATH, env=env)
    print("Model loaded successfully. Starting playback...")
    print("Close the PyBullet window to exit.")

    obs = env.reset()
    
    try:
        while True:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, infos = env.step(action)
            
            if done[0]:
                info = infos[0]
                success = info.get('success', False)
                print(f"Episode Finished. Success: {success} | Reward: {reward[0]:.2f}")
                time.sleep(1.0)
                obs = env.reset()
                
            # Slow down slightly for viewing
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    except (p.error, Exception) as e:
        # Catch the "Not connected" error if window is closed
        if "Not connected" in str(e):
            print("\nSimulation window closed. Exiting.")
        else:
            raise e
    finally:
        try:
            env.close()
        except:
            pass