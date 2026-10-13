"""Debug: co sie dzieje z katem tulowia w kazdym epizodzie?"""

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

obs = env.reset()
for ep in range(args.episodes):
    angles = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done[0]:
            break  # po done symulacja jest juz zresetowana
        angles.append(sim.data.qpos[2])
    angles = np.array(angles)
    deg = np.degrees(angles)
    upside = 100 * np.mean(np.cos(angles) < 0)
    print(
        f"ep {ep + 1:2d}: {info[0]['episode']['r']:6.0f} pkt | "
        f"kat: min {deg.min():5.0f}, max {deg.max():5.0f}, koniec {deg[-1]:5.0f} | "
        f"do gory nogami: {upside:3.0f}% krokow"
    )
