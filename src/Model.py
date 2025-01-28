from stable_baselines3 import PPO

import time

from DroneEnv import DroneEnv

# Create the environment
env = DroneEnv()

# Train the RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_drone")


# Load the trained model (optional)
# model = PPO.load("ppo_drone_model")


# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        time.sleep(10)
