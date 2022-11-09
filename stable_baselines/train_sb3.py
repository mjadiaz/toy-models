import gym
import toy_models

import numpy as np
from stable_baselines3 import TD3 

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


alg_name = 'TD3_double_gaussian_2'

env_config = TF2D_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.05
env_config.parameter_shift_mode = False
env_config.norm_min = -1
env_config.norm_max = 1
env_config.function_name = "double_gaussian"


env = gym.make("ToyFunction2d-v1")
#env = gym.make('Pendulum-v1')
if __name__ == '__main__':
    
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.4* np.ones(n_actions))
    
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./"+alg_name+"_toy_tensorboard/")
    
    model.learn(total_timesteps=120_000)
    
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
