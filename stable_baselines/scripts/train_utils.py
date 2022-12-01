import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gym
import toy_models
import wandb
import numpy as np
from stable_baselines3 import DDPG, TD3
from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from scripts.callbacks import TensorboardCallback
from wandb.integration.sb3 import WandbCallback
from omegaconf import DictConfig


def new_run(
        run_name: str, 
        env_name: str,
        total_timesteps: int, 
        env_config: DictConfig = None, 
        continue_training: bool = False
        ):

    final_model_name = "./logs/"+run_name+"/final"
    CONTINUE_TRAINING = continue_training 

    # Initiate env
    env = gym.make(env_name, env_config=env_config)

    # Initiate wandb
    run = wandb.init(
        project="sb3",
        config=dict(env_config),
        sync_tensorboard=True,
        )

    # Initiate noise objects for training strategy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))
    
    # Initiate the model
    if CONTINUE_TRAINING:
        model = TD3.load(final_model_name)
        model.set_env(env)
    else:
        model = TD3(
            "MlpPolicy", 
            env, 
            action_noise = action_noise, 
            verbose = 1,
            tensorboard_log = "./logs/"+run_name,
        )


    # Create Callbacks
    # Wandb
    wandb_callback=WandbCallback(verbose=2)
    
    # Custom Evaluation for custom metrics
    eval_env = gym.make(env_name, env_config=env_config)
    eval_env = Monitor(eval_env)
    eval_callback = TensorboardCallback(eval_env, n_eval_episodes=10, eval_freq=250, 
                                        deterministic=True, render=False)
   
    # Evaluation Callback for model saving 
    eval_env_2 = gym.make(env_name, env_config=env_config)
    eval_env_2 = Monitor(eval_env_2)
    eval_callback_save = EvalCallback(eval_env_2, best_model_save_path="./logs/"+run_name,
                             eval_freq=250,
                             deterministic=True, render=False)
    
    callback_list = CallbackList([eval_callback, wandb_callback, eval_callback_save])
    #  
    model.learn(
        total_timesteps=20_000,
        callback = callback_list
    )

    # Saving the final model
    model.save(final_model_name)

