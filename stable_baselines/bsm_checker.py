from stable_baselines3.common.env_checker import check_env
import gym
import bsm_models

from bsm_models.envs.bsm_environment import HEP_DEFAULT_CONFIG

hep_config = HEP_DEFAULT_CONFIG
hep_config.directories.reference_lhs = "/home/mjad1g20/HEP/SPHENO/modelfiles/LesHouches.in.BLSSM_high"



env = gym.make('PhenoEnv-v3', hep_config=hep_config)
# It will check your custom environment and output additional warnings if needed
#check_env(env)
state = env.reset()
print(state)
#
ns, r, done, info = env.step(env.action_space.sample())
#
##ns, r, done, info = env.step(env.action_space.sample())
print(ns)
print(r)
