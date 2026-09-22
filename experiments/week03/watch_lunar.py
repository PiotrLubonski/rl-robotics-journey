"""Obejrzyj wytrenowanego agenta."""

import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("runs/week03_ppo_lunar/ppo_lunar")
env = gym.make("LunarLander-v3", render_mode="human")

for ep in range(5):
    obs, info = env.reset()
    total = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            print(f"Epizod {ep + 1}: {total:.1f}")
            break

env.close()
