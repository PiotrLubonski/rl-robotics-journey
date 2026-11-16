"""Tydzień 10: SAC kontra ręczny regulator na SO101Reach (200 epizodów, te same cele)."""

from stable_baselines3 import SAC

from rl_journey.baselines import SO101JacobianController
from rl_journey.envs.so101_reach import SO101Reach
from rl_journey.eval import evaluate_env

MODELS = {
    "SAC 50 tys.": lambda: SAC.load("runs/week10/sac_so101_50000_seed0", device="cpu"),
    "SAC 150 tys.": lambda: SAC.load(
        "runs/week10/sac_so101_150000_seed0", device="cpu"
    ),
    "ręczny regulator": SO101JacobianController,
}

print(
    f"{'model':18s} {'sukces':>7s} {'nie domk.':>10s} {'nie traf.':>10s} {'koniec':>8s}"
)
for name, make in MODELS.items():
    r = evaluate_env(make(), SO101Reach())
    print(
        f"{name:18s} {r['success_pct']:6.1f}% {r['not_closed_pct']:9.1f}% "
        f"{r['missed_pct']:9.1f}% {r['end_cm']:5.1f} cm"
    )
