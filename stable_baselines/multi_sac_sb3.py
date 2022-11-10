import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import toy_models

import numpy as np
from stable_baselines3 import DDPG

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from callbacks import TensorboardCallback
from stable_baselines3.common.monitor import Monitor

alg_name = 'DDPG'

env_config = TF2D_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.3

'''
d_factor = range(1,10,0.5)
lh_factor = range(1,10,0.5)
kernel = range(0.1,0.7,0.1)
density_limit = range(0.1,0.5,0.1)
'''
if __name__ == '__main__':
    params = [
        (4, 2, 0.3), #d_factor, lh_factor, kernel
        (7, 2, 0.4),
        (4, 2, 0.5),
        (5, 2, 0.4),
        (5, 3, 0.3),
        ]
    for k in range(len(params)):
        d_factor, lh_factor, kernel = params[k]
        env_config.kernel_bandwith = kernel
        env_config.d_factor = d_factor 
        env_config.lh_factor = lh_factor
        env = gym.make("ToyFunction2d-v1")

        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", env, action_noise = action_noise, verbose = 1,
                        tensorboard_log = "./"+alg_name+"_toy_tensorboard/",)

        eval_env = gym.make('ToyFunction2d-v1')
        eval_env = Monitor(env)
        #eval_env = gym.make('Pendulum-v1')

        eval_callback = TensorboardCallback(eval_env, n_eval_episodes=10, 
                                            eval_freq=500, deterministic=True,
                                            render=False)

        model.learn(total_timesteps=10_000, callback = eval_callback)

        # Saving the final model
        model.save("test/"+alg_name+'_'+str(k))

    '''
    obs = env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
    '''
    env.close()


