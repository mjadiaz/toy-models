import gym
import toy_models

import numpy as np
from stable_baselines3 import TD3 

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.callbacks import EveryNTimesteps, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.logger import TensorBoardOutputFormat

alg_name = 'pendulum'

env_config = TF2D_DEFAULT_CONFIG
env_config.kernel_bandwidth = 0.05
env_config.parameter_shift_mode = False
env_config.norm_min = -1
env_config.norm_max = 1
env_config.function_name = "double_gaussian"

def make_env(env): 
    """
    See https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
    for more details on vectorized environments

    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init():
        return env
    return _init


#env = gym.make("ToyFunction2d-v1")
env_id = 'Pendulum-v1'
env = gym.make(env_id)
if __name__ == '__main__':
    # Define Callbacks 
    # this is equivalent to defining CheckpointCallback(save_freq=500)
    # checkpoint_callback will be triggered every 500 steps
    #checkpoint_on_event = CheckpointCallback(
    #        save_freq=1, 
    #        save_path='./logs/'+alg_name
    #        )
    #eval_callback_on_event = EvalCallback(
    #        env, 
    #        best_model_save_path='./logs/'+alg_name+'/best',
    #        log_path='./logs/'+alg_name, 
    #        eval_freq = 1
    #        )
    #callbacks = CallbackList([checkpoint_on_event, eval_callback_on_event])
    #callbacks = EveryNTimesteps(n_steps=500, callback=callbacks)
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3* np.ones(n_actions))
    
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./logs/"+alg_name)
    
    model.learn(total_timesteps=120_000, callback=callbacks)
    
    # Saving the final model
    #model.save("models/sac")
    model.save("./logs/"+alg_name+'/final')
    
    obs = env.reset()
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
    
    env.close()
