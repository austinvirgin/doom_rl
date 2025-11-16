# train_corridor.py
import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_MAX_THREADS"] = "10"

import gymnasium as gym
from vizdoom import gymnasium_wrapper  # registers envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("VizdoomCorridor-v0", render_mode=None)  # headless for speed/stability

model = PPO(
    "MultiInputPolicy",
    env,
    n_epochs=10,
    n_steps=1024,
    batch_size=256,
    verbose=1,
    tensorboard_log="./doom_logs"
)

# Save a checkpoint every 25k env steps
# ckpt_cb = CheckpointCallback(
#     save_freq=150,
#     save_path="./ckpts",
#     name_prefix="ppo_corridor"
# )

model.learn(total_timesteps=300_000, progress_bar=True)
model.save("ppo_corridor_final")
env.close()
