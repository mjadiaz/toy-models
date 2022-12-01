import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import gym
import numpy as np
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from scripts.custom_evaluation import custom_evaluate_policy

class TensorboardCallback(EventCallback):
    def __init__(
        self, 
        eval_env: Union[gym.Env, VecEnv], 
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5, 
        eval_freq: int = 500,
        #log_path: Optional[str] = None,
        #best_model_save_path: Optional[str] = None,
        deterministic: bool = True, 
        render: bool = False, 
        verbose: int = 1,
        warn: bool = True,
        ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env


    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" 
                f"{self.training_env} != {self.eval_env}")

        '''
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)
        '''

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                    ) from e

            episode_rewards, episode_length, area, precision  = custom_evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes, 
                render=self.render,
                deterministic=self.deterministic, 
                return_episode_rewards=True, 
                warn=self.warn
            )

            mean_area_covered = np.mean(area)
            mean_precision = np.mean(precision)
            mean_episode_rewards = np.mean(episode_rewards)
            mean_episode_length = np.mean(episode_length)
           
            self.logger.record('eval/rewards', mean_episode_rewards)
            self.logger.record('eval/episode_length', mean_episode_length)
            self.logger.record('eval/area_covered', mean_area_covered)
            self.logger.record('eval/precision', mean_precision)
            #self.logger.record(
            #        'eval/total_integration_avg', 
            #        total_integration_avg
            #        )

        return continue_training


