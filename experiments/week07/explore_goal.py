"""Tydzien 7: srodowisko z celami (goal-conditioned) i losowy agent."""

import argparse

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="FetchReach-v4")
parser.add_argument("--episodes", type=int, default=50)
args = parser.parse_args()

env = gym.make(args.env)
obs, info = env.reset(seed=0)

# 1. JAK WYGLADA OBSERWACJA
print("Obserwacja to slownik:")
for key, value in obs.items():
    print(f"  {key:14s} ksztalt {value.shape}  poczatek {np.round(value[:3], 3)}")
print("Akcje:", env.action_space)

# 2. LOSOWY AGENT: jak czesto trafia?
successes, final_dists = [], []
for ep in range(args.episodes):
    obs, info = env.reset(seed=ep)
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    successes.append(info["is_success"])
    final_dists.append(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

print(f"\nLosowy agent: sukces w {100 * np.mean(successes):.0f}% epizodow")
print(f"Srednia odleglosc od celu na koncu: {100 * np.mean(final_dists):.1f} cm")

# 3. SZTUCZKA HER: przelicz nagrode dla innego celu
achieved = obs["achieved_goal"]  # gdzie chwytak skonczyl ostatni epizod
desired = obs["desired_goal"]  # gdzie mial byc
r_real = env.unwrapped.compute_reward(achieved, desired, info)
r_relabel = env.unwrapped.compute_reward(achieved, achieved, info)
print(f"\nNagroda za prawdziwy cel:                     {r_real}")
print(
    f"Nagroda, gdyby celem bylo miejsce, gdzie skonczyl: {r_relabel}  (-0.0 to po prostu 0)"
)
