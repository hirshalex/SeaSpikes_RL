# run_test.py
from pybullet_ring_env import RingPickPlaceEnv
import time

if __name__ == "__main__":
    env = RingPickPlaceEnv(gui=True, num_tentacles=4, tentacle_curve=0.05, base_circle_radius=.30)
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Done:", info)
            env.reset()
    env.close()
