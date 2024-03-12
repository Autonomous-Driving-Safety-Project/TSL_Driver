import gymnasium as gym
import highway_env
import os
from stable_baselines3 import PPO, DQN

log_dir = "highway_dqn/DQN_2"

env = gym.make('highway-v0', render_mode='rgb_array')
# env.config["offscreen_rendering"] = False
# env.config["policy_frequency"] = 15
# env.config["action"]["type"] = "ContinuousAction"
# env.config["action"]["acceleration_range"] = (-10.0, 10.0)
# env.config["vehicles_density"] = 1
# env.config["offroad_terminal"] = True
env.reset()

model = DQN.load(os.path.join(log_dir,"model"))
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()