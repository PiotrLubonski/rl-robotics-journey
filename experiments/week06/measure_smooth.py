"""Pomiar plynnosci: kolysanie tulowia, szarpanie akcji, saturacja."""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

parser = argparse.ArgumentParser()
parser.add_argument("model", help="sciezka do modelu, bez .zip")
parser.add_argument("--vecnorm", default=None)
parser.add_argument("--env", default="HalfCheetah-v5")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1000)
args = parser.parse_args()

venv = make_vec_env(args.env, n_envs=1, seed=args.seed)
sim = venv.envs[0].unwrapped  # gola symulacja, zeby czytac kat
env = venv
if args.vecnorm:
    env = VecNormalize.load(args.vecnorm, venv)
    env.training = False
    env.norm_reward = False
model = PPO.load(args.model, device="cpu")

rewards, swings, jerks, sats = [], [], [], []
obs = env.reset()
for ep in range(args.episodes):
    angles, actions = [], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done[0]:
            break
        angles.append(sim.data.qpos[2])
        actions.append(np.clip(action[0], -1, 1))
    angles = np.degrees(np.array(angles))
    actions = np.array(actions)  # (kroki, 6)
    rewards.append(info[0]["episode"]["r"])
    swings.append(angles.std())  # kolysanie
    jerks.append(np.abs(np.diff(actions, axis=0)).mean())  # szarpanie
    sats.append(100 * np.mean(np.abs(actions) >= 0.99))  # saturacja

print(f"punkty:     {np.mean(rewards):7.0f}")
print(f"kolysanie:  {np.mean(swings):7.1f} st.")
print(f"szarpanie:  {np.mean(jerks):7.3f}  (0 = plynnie, 2 = skok z -1 na +1 co krok)")
print(f"saturacja:  {np.mean(sats):7.0f} %")
