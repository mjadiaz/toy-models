import gym
import bsm_models

import numpy as np
from stable_baselines3 import DDPG

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from bsm_checker import hep_config

alg_name = 'DDPG'

#env_config = TF2D_DEFAULT_CONFIG
#env_config.kernel_bandwidth = 0.1

env = gym.make("PhenoEnv-v3", hep_config=hep_config)
#env = gym.make('Pendulum-v1')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./"+alg_name+"_bsm_tensorboard/")

model.learn(total_timesteps=50_000)

# Saving the final model
#model.save("models/sac")
model.save("bsm/"+alg_name)


