from stable_baselines3.common.env_checker import check_env
import gym
import toy_models
import time
import numpy as np

env = gym.make('ToyFunction2d-v1', env_config=env_config)
# It will check your custom environment and output additional warnings if needed
check_env(env)

env.reset()
env.render()
for i in range(100):
    _,_,_,_ = env.step(env.action_space.sample())
    env.render()
    
