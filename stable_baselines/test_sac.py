import gym
import toy_models

from stable_baselines3 import SAC

env = gym.make("ToyFunction2d-v1")


model = SAC.load("models/sac")

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
