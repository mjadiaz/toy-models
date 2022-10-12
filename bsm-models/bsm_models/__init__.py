from gym.envs.registration import register

register(
    id='PhenoEnv-v3',
    entry_point='bsm_models.envs:PhenoEnv_v3',
)
