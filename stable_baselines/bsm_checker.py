from stable_baselines3.common.env_checker import check_env
import gym
import bsm_models

from bsm_models.envs.bsm_environment import HEP_DEFAULT_CONFIG

hep_config = HEP_DEFAULT_CONFIG
hep_config.directories.reference_lhs = "/home/mjad1g20/HEP/SPHENO/modelfiles/LesHouches.in.BLSSM_high"
hep_config.directories.higgsbounds = "/scratch/mjad1g20/HEP/higgsbounds-5.10.1/build"

hep_config.model.observation.name = ['Mh(1)', 'Mh(2)', 'obsratio']
hep_config.model.goal.name = ['Mh(1)', 'Mh(2)', 'obsratio']
hep_config.model.goal.value = [96, 125, 1]
hep_config.model.goal.lh_type= ['gaussian', 'gaussian', 'heaviside']


env = gym.make('PhenoEnv-v3')
# It will check your custom environment and output additional warnings if needed
#check_env(env)
state = env.reset()

ns, r, done, info = env.step(env.action_space.sample())

ns, r, done, info = env.step(env.action_space.sample())
print(ns)
print(r)
