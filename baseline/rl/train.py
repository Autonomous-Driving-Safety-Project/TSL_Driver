import gymnasium as gym
import highway_env
import os
from stable_baselines3 import PPO, DQN

log_dir = "highway_ppo/"

env = gym.make('highway-v0')
# env.config["offscreen_rendering"] = False
env.config["policy_frequency"] = 15
env.config["action"]["type"] = "ContinuousAction"
env.config["action"]["acceleration_range"] = (-10.0, 10.0)
env.config["vehicles_density"] = 1
env.config["offroad_terminal"] = True
env.reset()

# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log=log_dir)

model = PPO('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              verbose=1,
              tensorboard_log=log_dir)

model.learn(int(5e5))
model.save(os.path.join(log_dir,"model"))
env.close()