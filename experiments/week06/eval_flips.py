"""Ewaluacja z prawdziwym licznikiem przewrotek."""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wrappers import FlipMonitor

parser = argparse.ArgumentParser()
parser.add_argument("model", help="sciezka do modelu, bez .zip")
parser.add_argument("--vecnorm", default=None, help="plik _vecnormalize.pkl")
parser.add_argument("--env", default="HalfCheetah-v5")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)  # te same starty co w tygodniu 5
args = parser.parse_args()

env = make_vec_env(args.env, n_envs=1, seed=args.seed, wrapper_class=FlipMonitor)
if args.vecnorm:
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

model = PPO.load(args.model, device="cpu")

rewards, flips = [], []
obs = env.reset()
while len(rewards) < args.episodes:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = env.step(action)
    if done[0]:  # koniec epizodu
        rewards.append(info[0]["episode"]["r"])  # prawdziwe punkty z Monitora
        flips.append(info[0]["flipped"])  # nasz wrapper

rewards = np.array(rewards)
flips = np.array(flips)
slow = (rewards < 1000) & ~flips  # slabe, ale bez przewrotki

print(f"srednia: {rewards.mean():.1f}  ({len(rewards)} epizodow)")
print(f"przewrotki: {flips.sum()} / {len(flips)}  ({100 * flips.mean():.0f}%)")
print(f"wolne bez przewrotki (<1000): {slow.sum()}")
if (~flips).any():
    print(f"srednia bez przewrotki: {rewards[~flips].mean():.1f}")
