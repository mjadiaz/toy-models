import gym
import toy_models

from stable_baselines3 import PPO

#env = gym.make("ToyFunction2d-v1")
env = gym.make('Pendulum-v1')

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./sac_toy_tensorboard/")
model.learn(total_timesteps=100_000)

# Saving the final model
#model.save("models/sac")
model.save("test/PPO")

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
