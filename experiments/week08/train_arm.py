"""Tydzień 8: SAC na ArmReach + ocena na 50 epizodach."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from rl_journey.envs.arm_reach import ArmReach
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

RUN = Path("runs/week08")
RESULTS = RUN / "results.csv"


def evaluate(model, env_kwargs, episodes=50, seed=1000):
    env = ArmReach(**env_kwargs)
    end, ever = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        hit = False
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            hit = hit or info["is_success"] == 1.0
        end.append(info["distance"])
        ever.append(hit)
    end = np.array(end)
    return {
        "end_cm": round(100 * end.mean(), 1),
        "success_pct": round(100 * (end < env.threshold).mean(), 1),
        "ever_pct": round(100 * np.mean(ever), 1),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--angle-obs", choices=["cossin", "raw"], default="cossin")
parser.add_argument("--no-velocity", action="store_true")
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env_kwargs = dict(angle_obs=args.angle_obs, with_velocity=not args.no_velocity)
name = f"sac_arm_{args.angle_obs}_vel{int(not args.no_velocity)}_{args.steps}_seed{args.seed}"
RUN.mkdir(parents=True, exist_ok=True)

model = SAC(
    "MlpPolicy",
    Monitor(ArmReach(**env_kwargs)),
    seed=args.seed,
    verbose=1,
    device="cpu",
    tensorboard_log=str(RUN / "tb"),
)
start = time.perf_counter()
model.learn(total_timesteps=args.steps, tb_log_name=name, log_interval=10)
minutes = (time.perf_counter() - start) / 60
model.save(RUN / name)

row = {
    "angle_obs": args.angle_obs,
    "velocity": int(not args.no_velocity),
    "steps": args.steps,
    "seed": args.seed,
    **evaluate(model, env_kwargs),
    "minutes": round(minutes, 1),
}
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(
    f"{name}: koniec {row['end_cm']} cm | sukces na koncu {row['success_pct']}% | "
    f"dotknal choc raz {row['ever_pct']}% | {row['minutes']} min"
)
