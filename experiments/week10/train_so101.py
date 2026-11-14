"""Tydzień 10: SAC na SO101Reach (model SO-101 w MuJoCo)."""

import argparse
import csv
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl_journey.envs.so101_reach import SO101Reach
from rl_journey.eval import evaluate_env

RUN = Path("runs/week10")
RESULTS = RUN / "results.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

name = f"sac_so101_{args.steps}_seed{args.seed}"
RUN.mkdir(parents=True, exist_ok=True)

model = SAC(
    "MlpPolicy",
    Monitor(SO101Reach()),
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
    "steps": args.steps,
    "seed": args.seed,
    **evaluate_env(model, SO101Reach()),
    "minutes": round(minutes, 1),
}
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(
    f"{name}: sukces {row['success_pct']}% | nie domknal {row['not_closed_pct']}% | "
    f"nie trafil {row['missed_pct']}% | koniec {row['end_cm']} cm | {row['minutes']} min"
)
