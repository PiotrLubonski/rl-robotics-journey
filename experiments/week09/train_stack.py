"""Tydzień 9: SAC na ArmReach bez prędkości, z pamięcią n ostatnich obserwacji."""

import argparse
import csv
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.eval import evaluate_arm
from rl_journey.wrappers import StackObs

RUN = Path("runs/week09")
RESULTS = RUN / "results_stack.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--stack", type=int, default=2)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

name = f"sac_arm_novel_stack{args.stack}_{args.steps}_seed{args.seed}"
RUN.mkdir(parents=True, exist_ok=True)

env_kwargs = {"with_velocity": False}


def wrap(e):
    return StackObs(e, n=args.stack)


env = Monitor(wrap(ArmReach(**env_kwargs)))

model = SAC(
    "MlpPolicy",
    env,
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
    "stack": args.stack,
    "steps": args.steps,
    "seed": args.seed,
    **evaluate_arm(model, env_kwargs=env_kwargs, wrapper=wrap),
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
    f"nie trafil {row['missed_pct']}% | koniec {row['end_cm']} cm | "
    f"kraniec {row['edge_success_pct']}% (n={row['edge_n']}) | {row['minutes']} min"
)
