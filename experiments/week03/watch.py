"""Obejrzyj wybrany model z eksperymentu."""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="sciezka do modelu, bez .zip")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = gym.make("LunarLander-v3", render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset()
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                end = "limit czasu" if truncated else "koniec"
                print(f"Epizod {ep + 1}: {total:.1f} pkt, {steps} krokow, {end}")
                break

    env.close()


if __name__ == "__main__":
    main()
