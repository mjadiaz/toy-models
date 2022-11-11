import gym
import bsm_models

import numpy as np
from stable_baselines3 import DDPG, TD3
from bsm_models.envs.bsm_environment import PHENOENV_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.callbacks import CheckpointCallback


from bsm_checker import hep_config

alg_name = 'TD3'

env_config =  PHENOENV_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.05
env_config.parameter_shift_mode = False
env_config.norm_min = -1
env_config.norm_max = 1

env = gym.make("PhenoEnv-v3", env_config=env_config, hep_config=hep_config)
#env = gym.make('Pendulum-v1')

# The noise objects for DDPG
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./"+alg_name+"_bsm_tensorboard/")

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./logs/',
                                         name_prefix='rl_model')
model.learn(total_timesteps=100_000,callback=checkpoint_callback)

# Saving the final model
#model.save("models/sac")
model.save("bsm/"+alg_name)


