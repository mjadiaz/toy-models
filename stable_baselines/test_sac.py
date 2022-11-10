import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import toy_models

from stable_baselines3 import DDPG
from sac_sb3 import env_config

env = gym.make("ToyFunction2d-v1", env_config = env_config)


model = DDPG.load("test/DDPG")

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
