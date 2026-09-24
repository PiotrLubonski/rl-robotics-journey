"""Nagraj GIF z wytrenowanym agentem."""

import argparse
from pathlib import Path

import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="sciezka do modelu, bez .zip")
    parser.add_argument("--out", default="media/lunar_ppo.gif")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    obs, info = env.reset(seed=args.seed)
    frames, total = [], 0.0
    while True:
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()

    images = [Image.fromarray(f).resize((300, 200)) for f in frames[::2]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=40, loop=0, optimize=True
    )
    print(f"Zapisano {out}: {len(images)} klatek, wynik epizodu {total:.1f}")


if __name__ == "__main__":
    main()
