from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
from scripts.train_utils import new_run
import os

wandb_key = os.environ.get('WANDB_KEY')


run_name = 'test' 
env_name = 'ToyFunction2d-v1'
continue_training = False

env_config = TF2D_DEFAULT_CONFIG
env_config.d_factor = 10
env_config.lh_factor = 5
env_config.kernel_bandwidth = 0.2
env_config.density_limit = 0.6

new_run(
    run_name,
    env_name,
    total_timesteps=2000,
    env_config = env_config,
    cluster_mode = True,
    wandb_key = wandb_key,
    )
