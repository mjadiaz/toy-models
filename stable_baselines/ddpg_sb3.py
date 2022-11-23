import gym
import bsm_models

import numpy as np
from stable_baselines3 import DDPG, TD3
from bsm_models.envs.bsm_environment import PHENOENV_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback


from bsm_checker import hep_config

alg_name = 'blssm_td3_1'

env_config =  PHENOENV_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.1
env_config.parameter_shift_mode = False
env_config.norm_min = -1
env_config.norm_max = 1

env = gym.make("PhenoEnv-v3", env_config=env_config, hep_config=hep_config)

eval_env = gym.make("PhenoEnv-v3", env_config=env_config, hep_config=hep_config)

#env = gym.make('Pendulum-v1')

# The noise objects for DDPG
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./logs/"+alg_name)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./logs/'+alg_name+'/chckpts/',
                                         name_prefix='model')

# Separate evaluation env
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/'+alg_name+'/results/best_model',
                             log_path='./logs/'+alg_name+'/results', eval_freq=100)
# Create the callback list
callback = CallbackList([checkpoint_callback, eval_callback])
model.learn(total_timesteps=10_000,callback=callback)

# Saving the final model
#model.save("models/sac")
model.save("bsm/"+alg_name)


