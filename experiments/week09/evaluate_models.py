"""Ocena zapisanych modeli ArmReach tym samym sprawdzianem (200 epizodów, seedy 1000+)."""

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from rl_journey.eval import evaluate_arm
from rl_journey.wrappers import StackObs

parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="+", help="ścieżki do modeli (bez .zip)")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument(
    "--no-velocity", action="store_true", help="ramię bez prędkości w obserwacji"
)
parser.add_argument(
    "--stack", type=int, default=1, help="ile ostatnich obserwacji (1 = bez pamięci)"
)
args = parser.parse_args()

env_kwargs = {"with_velocity": not args.no_velocity}
wrapper = (lambda e: StackObs(e, n=args.stack)) if args.stack > 1 else None

print(
    f"{'model':36s} {'sukces':>7s} {'nie domk.':>10s} {'nie traf.':>10s} "
    f"{'koniec':>8s} {'kraniec (n)':>13s}"
)
for path in args.models:
    r = evaluate_arm(
        SAC.load(path, device="cpu"),
        env_kwargs=env_kwargs,
        episodes=args.episodes,
        wrapper=wrapper,
    )
    print(
        f"{Path(path).name:36s} {r['success_pct']:6.1f}% {r['not_closed_pct']:9.1f}% "
        f"{r['missed_pct']:9.1f}% {r['end_cm']:5.1f} cm {r['edge_success_pct']:6.1f}% ({r['edge_n']})"
    )
