"""Tydzien 4: poznaj srodowiska z akcjami ciaglymi."""

import gymnasium as gym
import numpy as np


def describe(env_id, episodes=10):
    env = gym.make(env_id)
    print(f"\n=== {env_id} ===")
    print("obserwacja:", env.observation_space)
    print("akcja:     ", env.action_space)
    print("przykladowa akcja:", env.action_space.sample())

    returns = []
    for ep in range(episodes):
        env.reset(seed=ep)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            if terminated or truncated:
                break
        returns.append(total)

    end = "terminated" if terminated else "truncated"
    print(f"losowy agent: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"epizody koncza sie przez: {end}")
    env.close()


def show_random(env_id, episodes=3):
    env = gym.make(env_id, render_mode="human")
    for ep in range(episodes):
        env.reset(seed=ep)
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
    env.close()


if __name__ == "__main__":
    describe("Pendulum-v1")
    describe("Reacher-v5")
    show_random("Reacher-v5")
