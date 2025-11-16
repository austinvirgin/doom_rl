# watch_all_checkpoints.py
import os, glob, time
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # registers VizDoom envs
from stable_baselines3 import PPO
import vizdoom

BASE_DIR = r"C:\Users\austi\Documents\Programming\doom_rl"
CKPT_DIR = os.path.join(BASE_DIR, "ckpts")

def list_ckpts():
    return sorted(glob.glob(os.path.join(CKPT_DIR, "ppo_corridor_*.zip")), key=os.path.getmtime)

while True:
    ckpts = list_ckpts()
    if not ckpts:
        print("ViZDoom version:", vizdoom.__version__)
        print("✅ No more checkpoints left.")
        break

    ckpt = ckpts[0]             # oldest first; use ckpts[-1] for newest first
    print(f"▶ Loading: {ckpt}")

    env = gym.make("VizdoomCorridor-v0", render_mode="human")
    try:
        model = PPO.load(ckpt, env=env, print_system_info=False)
        obs, info = env.reset()
        total = 0.0
        for _ in range(100):    # ~20–30 seconds
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, trunc, info = env.step(action)
            total += r
            if done or trunc:
                obs, info = env.reset()
        print(f"✅ Finished {os.path.basename(ckpt)} | total_reward={total:.1f}")
    finally:
        env.close()

    # delete after viewing
    try:
        os.remove(ckpt)
        print(f"🗑 Deleted: {ckpt}")
    except Exception as e:
        print(f"⚠ Could not delete {ckpt}: {e}")

    time.sleep(0.2)
