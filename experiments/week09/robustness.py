"""Tydzień 9: odporność polityki na zmienioną fizykę i szum (200 epizodów na scenariusz)."""

import argparse
import csv
from pathlib import Path

from stable_baselines3 import SAC

from rl_journey.baselines import JacobianController
from rl_journey.eval import evaluate_arm
from rl_journey.wrappers import ObsNoise, RandomizePhysics

BEZ_ZMIAN = (1.0, 1.0)
SCENARIOS = {
    "nominalny": None,
    "dlugosci ±10%": lambda e: RandomizePhysics(
        e, damping_scale=BEZ_ZMIAN, gain_scale=BEZ_ZMIAN
    ),
    "tarcie ×0,5–2": lambda e: RandomizePhysics(
        e, link_scale=BEZ_ZMIAN, gain_scale=BEZ_ZMIAN
    ),
    "silnik ±30%": lambda e: RandomizePhysics(
        e, link_scale=BEZ_ZMIAN, damping_scale=BEZ_ZMIAN
    ),
    "szum 2 cm": lambda e: ObsNoise(e, std=0.02),
    "wszystko naraz": lambda e: ObsNoise(RandomizePhysics(e), std=0.02),
}
RESULTS = Path("runs/week09/robustness.csv")

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", help="ścieżka do modelu SAC (bez .zip)")
parser.add_argument(
    "--hand", action="store_true", help="oceń ręczny regulator zamiast modelu"
)
parser.add_argument("--episodes", type=int, default=200)
args = parser.parse_args()

if args.hand:
    model, name = JacobianController(), "reczny_regulator"
else:
    model, name = SAC.load(args.model, device="cpu"), Path(args.model).name
print(f"model: {name}")
print(
    f"{'scenariusz':16s} {'sukces':>7s} {'nie domk.':>10s} {'nie traf.':>10s} "
    f"{'koniec':>8s} {'kraniec (n)':>13s}"
)
rows = []
for scenario, wrapper in SCENARIOS.items():
    r = evaluate_arm(model, episodes=args.episodes, wrapper=wrapper)
    rows.append({"model": name, "scenariusz": scenario, **r})
    print(
        f"{scenario:16s} {r['success_pct']:6.1f}% {r['not_closed_pct']:9.1f}% "
        f"{r['missed_pct']:9.1f}% {r['end_cm']:5.1f} cm {r['edge_success_pct']:6.1f}% ({r['edge_n']})"
    )

RESULTS.parent.mkdir(parents=True, exist_ok=True)
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new_file:
        writer.writeheader()
    writer.writerows(rows)
