"""Tydzień 10: gdzie są porażki? Sukces według odległości celu od osi i wysokości (200 epizodów)."""

import numpy as np
from stable_baselines3 import SAC

from rl_journey.baselines import SO101JacobianController
from rl_journey.envs.so101_reach import SO101Reach

MODELS = {
    "ręczny regulator": SO101JacobianController,
    "SAC 150 tys.": lambda: SAC.load(
        "runs/week10/sac_so101_150000_seed0", device="cpu"
    ),
}
RADIUS = [
    (0.12, 0.20, "blisko 12–20 cm"),
    (0.20, 0.28, "średnio 20–28 cm"),
    (0.28, 0.39, "daleko 28–38 cm"),
]
HEIGHT = [
    (0.00, 0.10, "nisko 2–10 cm"),
    (0.10, 0.20, "środek 10–20 cm"),
    (0.20, 0.31, "wysoko 20–30 cm"),
]


def run(model, env, episodes=200, seed=1000):
    """Dla każdego epizodu: seed, odległość celu od osi, wysokość celu, odległość na końcu, sukces."""
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        radius = float(np.linalg.norm(env.goal[:2] - env.base_xy))
        height = float(env.goal[2])
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
        rows.append((seed + ep, radius, height, info["distance"], info["is_success"]))
    return np.array(rows)


def by_group(rows, col, groups, title):
    print(f"  {title}:")
    for lo, hi, label in groups:
        mask = (rows[:, col] >= lo) & (rows[:, col] < hi)
        if mask.sum():
            print(
                f"    {label:18s} sukces {100 * rows[mask, 4].mean():5.1f}%  (n={mask.sum()})"
            )


env = SO101Reach()
for name, make in MODELS.items():
    rows = run(make(), env)
    print(f"\n{name}: sukces {100 * rows[:, 4].mean():.1f}%")
    by_group(rows, 1, RADIUS, "odległość celu od osi podstawy")
    by_group(rows, 2, HEIGHT, "wysokość celu nad stołem")
    fails = rows[rows[:, 4] == 0]
    print(f"  porażki: {len(fails)}; pierwsze 10 (seed | od osi | wysokość | koniec):")
    for s, r, h, d, _ in fails[:10]:
        print(
            f"    {int(s)} | {100 * r:5.1f} cm | {100 * h:5.1f} cm | {100 * d:5.1f} cm"
        )
