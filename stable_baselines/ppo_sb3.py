import gym
import toy_models

from stable_baselines3 import SAC

env = gym.make("ToyFunction2d-v1")

model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./sac_toy_tensorboard/")
model.learn(total_timesteps=30_000)

# Saving the final model
model.save("models/sac")

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
