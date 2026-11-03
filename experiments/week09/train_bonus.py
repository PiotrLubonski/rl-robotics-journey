"""Tydzień 9: SAC na ArmReach z premią za bycie w progu."""

import argparse
import csv
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.eval import evaluate_arm
from rl_journey.wrappers import SuccessBonus

RUN = Path("runs/week09")
RESULTS = RUN / "results.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--bonus", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

name = f"sac_arm_bonus{args.bonus:g}_{args.steps}_seed{args.seed}".replace(".", "p")
RUN.mkdir(parents=True, exist_ok=True)

# Monitor najbliżej ramienia zapisuje ORYGINALNĄ nagrodę (bez premii),
# więc ep_rew_mean da się porównać z tygodniem 8. SAC dostaje nagrodę z premią.
env = SuccessBonus(Monitor(ArmReach()), bonus=args.bonus)

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
    "bonus": args.bonus,
    "steps": args.steps,
    "seed": args.seed,
    **evaluate_arm(model),
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
