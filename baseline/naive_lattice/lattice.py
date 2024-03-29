import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from planner.lattice_planner import lattice_plan, get_surround_vehicles
from planner.pure_persuit import pure_persuit
import gymnasium as gym
import pandas as pd
import numpy as np
from datetime import datetime

env = gym.make('highway-v0', render_mode='rgb_array')
env.config["offscreen_rendering"] = False
env.config["policy_frequency"] = 15
env.config["action"]["type"] = "ContinuousAction"
env.config["action"]["acceleration_range"] = (-10.0, 10.0)
env.config["vehicles_density"] = 1

def plan_without_model(road, ego):
    traj = lattice_plan(road, ego, 30.0, obstacles=get_surround_vehicles(road, ego))
    if traj is None:
        return None
    acc, steer = pure_persuit(traj, ego)
    action = np.array([acc / 10.0, steer / 0.7853981633974483])
    return action

for episode in range(50):
    env.reset()
    env.render()
    metrics = pd.DataFrame(columns=["frame", "speed", "TTC_front", "TTC_rear", "done", "truncated"])
    frame_count = 0
    
    while True:
        print("----------------------------------------------")
        road, ego = env.unwrapped.get_ground_truth()
        action = None
        action = plan_without_model(road, ego)
        if action is None:
            print("Unable to plan. Use zero action.")
            action = np.array([0.0, 0.0])
        obs, reward, done, truncated, info = env.step(action)
        metric = env.unwrapped.get_metrics()
        print(metric)
        metrics = pd.concat([metrics,
            pd.DataFrame({
                "frame": frame_count,
                "speed": metric["speed"],
                "TTC_front": metric["TTC_front"],
                "TTC_rear": metric["TTC_rear"],
                "done": done,
                "truncated": truncated
            },index=[0])], ignore_index=True
        )
        frame_count += 1
        env.render()
        
        if done or truncated:
            break

    metrics.to_csv(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')}metrics_{episode}.csv")

input("Press Enter to continue...")
env.close()