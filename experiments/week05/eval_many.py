"""Ewaluacja na wielu epizodach: ile udanych biegow, ile przewrotek."""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

parser = argparse.ArgumentParser()
parser.add_argument("model", help="sciezka do modelu, bez .zip")
parser.add_argument("--vecnorm", default=None, help="plik _vecnormalize.pkl")
parser.add_argument("--env", default="HalfCheetah-v5")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)
# przewrotka daje ~600 pkt, kazdy bieg (nawet wolny, ~1600) jest powyzej 1000
parser.add_argument("--fail", type=float, default=1000, help="ponizej = porazka")
args = parser.parse_args()

env = make_vec_env(args.env, n_envs=1, seed=args.seed)
if args.vecnorm:
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False  # nie aktualizuj statystyk
    env.norm_reward = False  # chcemy prawdziwe punkty

model = PPO.load(args.model, device="cpu")
rewards, _ = evaluate_policy(
    model, env, n_eval_episodes=args.episodes, return_episode_rewards=True
)
rewards = np.array(rewards)
fails = rewards < args.fail  # True/False, jedno na epizod

print(
    f"srednia: {rewards.mean():.1f} +/- {rewards.std():.1f}  ({len(rewards)} epizodow)"
)
print(f"porazki: {fails.sum()} / {len(rewards)}  ({100 * fails.mean():.0f}%)")
if (~fails).any():
    print(f"srednia udanych: {rewards[~fails].mean():.1f}")
if fails.any():
    print(f"srednia porazek: {rewards[fails].mean():.1f}")
