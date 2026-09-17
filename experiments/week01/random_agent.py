"""Baseline: random agent on CartPole-v1."""

import gymnasium as gym
import numpy as np

N_EPISODES = 100


def run_episode(env, seed=None):
    """Run one episode with random actions. Return (n_steps, terminated, truncated)."""
    obs, info = env.reset(seed=seed)
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            return steps, terminated, truncated


def main():
    env = gym.make("CartPole-v1")

    lengths = []
    n_terminated = 0
    n_truncated = 0

    for i in range(N_EPISODES):
        steps, terminated, truncated = run_episode(env, seed=i)
        lengths.append(steps)
        if truncated:
            n_truncated += 1
        if terminated:
            n_terminated += 1

    env.close()

    lengths = np.array(lengths)
    print(f"Episodes:  {N_EPISODES}")
    print(f"Mean:      {lengths.mean():.1f}")
    print(f"Std:       {lengths.std():.1f}")
    print(f"Min / Max: {lengths.min()} / {lengths.max()}")
    print(f"Terminated: {n_terminated}, Truncated: {n_truncated}")


if __name__ == "__main__":
    main()
