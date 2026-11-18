"""Tydzień 10: gdzie są porażki? Sukces według odległości od osi, wysokości i kierunku celu (200 epizodów)."""

import mujoco
import numpy as np
from stable_baselines3 import SAC

from rl_journey.baselines import SO101JacobianController
from rl_journey.envs.so101_reach import SO101Reach

MODELS = {
    "ręczny regulator": SO101JacobianController,
    "SAC 150 tys.": lambda: SAC.load(
        "runs/week10/sac_so101_150000_seed0", device="cpu"
    ),
    "SAC 300 tys.": lambda: SAC.load(
        "runs/week10/sac_so101_300000_seed0", device="cpu"
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
TURN = [
    (0, 70, "przodem (obrót < 70°)"),
    (70, 110.5, "z boku (70–110°)"),
    (110.5, 181, "za plecami (> 110°)"),
]


def azimuth(env, xyz):
    """Kierunek punktu widziany z góry, od osi podstawy, w stopniach (−180…180)."""
    d = xyz[:2] - env.base_xy
    return np.degrees(np.arctan2(d[1], d[0]))


def turn_table(env):
    """Obracamy podstawę co 1° przy pozostałych stawach na 0 (ramię wyciągnięte przodem)
    i zapisujemy, w którą stronę wtedy celuje chwytak."""
    turns = np.arange(-180, 181)
    directions = []
    for t in turns:
        env.fk_data.qpos[env.qpos_adr] = [np.radians(t), 0, 0, 0, 0]
        mujoco.mj_kinematics(env.model, env.fk_data)
        directions.append(azimuth(env, env.fk_data.site_xpos[env.tip_id]))
    return turns, np.array(directions)


def needed_turn(env, table):
    """O ile stopni trzeba obrócić podstawę, żeby ramię wyciągnięte przodem celowało w cel."""
    turns, directions = table
    diff = (
        directions - azimuth(env, env.goal) + 180
    ) % 360 - 180  # różnica kątów zawinięta do −180…180
    return abs(turns[np.argmin(np.abs(diff))])


def run(model, env, table, episodes=200, seed=1000):
    """Dla każdego epizodu: seed, od osi, wysokość, potrzebny obrót podstawy, odległość na końcu, sukces."""
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        radius = float(np.linalg.norm(env.goal[:2] - env.base_xy))
        height = float(env.goal[2])
        turn = float(needed_turn(env, table))
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
        rows.append(
            (seed + ep, radius, height, turn, info["distance"], info["is_success"])
        )
    return np.array(rows)


def by_group(rows, col, groups, title):
    print(f"  {title}:")
    for lo, hi, label in groups:
        mask = (rows[:, col] >= lo) & (rows[:, col] < hi)
        if mask.sum():
            print(
                f"    {label:24s} sukces {100 * rows[mask, 5].mean():5.1f}%  (n={mask.sum()})"
            )


env = SO101Reach()
table = turn_table(env)
for name, make in MODELS.items():
    rows = run(make(), env, table)
    print(f"\n{name}: sukces {100 * rows[:, 5].mean():.1f}%")
    by_group(rows, 1, RADIUS, "odległość celu od osi podstawy")
    by_group(rows, 2, HEIGHT, "wysokość celu nad stołem")
    by_group(rows, 3, TURN, "kierunek celu")
    fails = rows[rows[:, 5] == 0]
    print(f"  porażki: {len(fails)} (seed | od osi | wysokość | obrót | koniec):")
    for s, r, h, t, d, _ in fails[:15]:
        print(
            f"    {int(s)} | {100 * r:5.1f} cm | {100 * h:5.1f} cm | {t:4.0f}° | {100 * d:5.1f} cm"
        )
