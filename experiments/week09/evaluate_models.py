"""Ocena zapisanych modeli ArmReach tym samym sprawdzianem (200 epizodów, seedy 1000+)."""

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from rl_journey.eval import evaluate_arm

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+", help="ścieżki do modeli (bez .zip)")
parser.add_argument("--episodes", type=int, default=200)
args = parser.parse_args()

print(
    f"{'model':36s} {'sukces':>7s} {'nie domk.':>10s} {'nie traf.':>10s} "
    f"{'koniec':>8s} {'kraniec (n)':>13s}"
)
for path in args.models:
    r = evaluate_arm(SAC.load(path, device="cpu"), episodes=args.episodes)
    print(
        f"{Path(path).name:36s} {r['success_pct']:6.1f}% {r['not_closed_pct']:9.1f}% "
        f"{r['missed_pct']:9.1f}% {r['end_cm']:5.1f} cm {r['edge_success_pct']:6.1f}% ({r['edge_n']})"
    )
