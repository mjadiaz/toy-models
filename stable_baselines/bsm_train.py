from bsm_models.envs.bsm_environment import PHENOENV_DEFAULT_CONFIG
from bsm_models.envs.bsm_environment import HEP_DEFAULT_CONFIG 

from scripts.train_utils_bsm import new_run
import os

wandb_key = os.environ.get('WANDB_KEY')


run_name = 'blssm_2' 
env_name = 'PhenoEnv-v3'
continue_training = False

env_config = PHENOENV_DEFAULT_CONFIG 
env_config.d_factor = 10
env_config.lh_factor = 5
env_config.kernel_bandwidth = 0.1

hep_config = HEP_DEFAULT_CONFIG
hep_config.directories.reference_lhs = "/home/mjad1g20/HEP/SPHENO/modelfiles/LesHouches.in.BLSSM_high"
hep_config.directories.higgssignals = "/home/mjad1g20/HEP/HS/higgssignals-2.6.2/build"
hep_config.directories.higgsbounds = "/home/mjad1g20/HEP/HB/higgsbounds-5.10.2/build"

new_run(
    run_name=run_name,
    env_name=env_name,
    total_timesteps=3000,
    env_config = env_config,
    hep_config = hep_config,
    cluster_mode = True,
    wandb_key = wandb_key,
    project_name = 'BLSSM'
    )
