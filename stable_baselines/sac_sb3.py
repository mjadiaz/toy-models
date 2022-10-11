import gym
import toy_models

import numpy as np
from stable_baselines3 import DDPG

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


alg_name = 'DDPG'

env_config = TF2D_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.1

env = gym.make("ToyFunction2d-v1")
#env = gym.make('Pendulum-v1')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./"+alg_name+"_toy_tensorboard/")

model.learn(total_timesteps=50_000)

# Saving the final model
#model.save("models/sac")
model.save("test/"+alg_name)

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
