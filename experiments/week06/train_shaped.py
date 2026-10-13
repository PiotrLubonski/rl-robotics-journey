"""Tydzien 6: domyslne PPO + kara za przechylenie tulowia."""

# 1. NARZEDZIA
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wrappers import FlipMonitor, UprightReward

RUN = Path("runs/week06")
RESULTS = RUN / "results.csv"

# 2. USTAWIENIA
parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=float, default=1.0, help="sila kary, 0 = bez kary")
parser.add_argument("--env", default="HalfCheetah-v5")
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-episodes", type=int, default=50)
args = parser.parse_args()

RUN.mkdir(parents=True, exist_ok=True)
name = f"ppo_upright{args.weight}_seed{args.seed}"

# 3. TRENING: PPO widzi nagrode z kara.
# Monitor siedzi pod spodem, wiec zapisuje oryginalne punkty (bez kary).
env = make_vec_env(
    args.env,
    n_envs=1,
    seed=args.seed,
    wrapper_class=UprightReward,
    wrapper_kwargs=dict(weight=args.weight),
)  # argumenty dla __init__
model = PPO(
    "MlpPolicy",
    env,
    seed=args.seed,
    verbose=1,
    device="cpu",
    tensorboard_log=str(RUN / "tb"),
)
start = time.perf_counter()
model.learn(total_timesteps=args.steps, tb_log_name=name)
minutes = (time.perf_counter() - start) / 60
model.save(RUN / name)

# 4. OCENA: oryginalna nagroda + licznik przewrotek, te same 50 startow co zawsze
eval_env = make_vec_env(args.env, n_envs=1, seed=1000, wrapper_class=FlipMonitor)
rewards, flips = [], []
obs = eval_env.reset()
while len(rewards) < args.eval_episodes:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = eval_env.step(action)
    if done[0]:
        rewards.append(info[0]["episode"]["r"])
        flips.append(info[0]["flipped"])
rewards = np.array(rewards)
flips = np.array(flips)

# 5. ZAPIS
row = {
    "weight": args.weight,
    "seed": args.seed,
    "mean": round(float(rewards.mean()), 1),
    "flip_pct": round(100 * float(flips.mean())),
    "slow": int(((rewards < 1000) & ~flips).sum()),
    "minutes": round(minutes, 1),
}
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(
    f"{name}: srednia {row['mean']}, na plecach {row['flip_pct']}%, "
    f"wolne {row['slow']} ({row['minutes']} min)"
)
