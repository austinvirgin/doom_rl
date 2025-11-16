import gymnasium as gym
from vizdoom import gymnasium_wrapper  # registers envs
from stable_baselines3 import PPO

# Load
model = PPO.load("ppo_corridor_final")  # CPU/GPU auto-detected

# Human render so you can see it
env = gym.make("VizdoomCorridor-v0", render_mode="human")

obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()