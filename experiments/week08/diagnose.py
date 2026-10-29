"""Diagnoza: w które cele ramię trafia, a w które nie."""

import argparse

import numpy as np
from arm_reach import ArmReach
from stable_baselines3 import SAC

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--angle-obs", choices=["cossin", "raw"], default="cossin")
parser.add_argument("--no-velocity", action="store_true")
parser.add_argument("--episodes", type=int, default=200)
args = parser.parse_args()

model = SAC.load(args.model, device="cpu")
env = ArmReach(angle_obs=args.angle_obs, with_velocity=not args.no_velocity)

radius, start, end, closest, group = [], [], [], [], []
for ep in range(args.episodes):
    obs, info = env.reset(seed=1000 + ep)
    radius.append(np.linalg.norm(env.target))  # jak daleko cel od barku
    start.append(
        np.linalg.norm(env.fingertip(env.q) - env.target)
    )  # jak daleko było na starcie
    dists = []
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        dists.append(info["distance"])
    end.append(dists[-1])
    closest.append(min(dists))
    if dists[-1] < env.threshold:
        group.append("sukces")
    elif min(dists) < env.threshold:
        group.append("nie domknal")
    else:
        group.append("nie trafil")

radius, start, end, closest, group = map(np.array, (radius, start, end, closest, group))
names = ["sukces", "nie domknal", "nie trafil"]

print(f"\n=== Grupy ({args.episodes} epizodow) ===")
for g in names:
    m = group == g
    if m.any():
        print(
            f"{g:12s} {m.mean():4.0%} | koniec sr. {100 * end[m].mean():5.1f} cm | "
            f"najblizej sr. {100 * closest[m].mean():5.1f} cm"
        )
    else:
        print(f"{g:12s}   0%")


def table(title, values, edges):
    print(f"\n=== {title} ===")
    print(f"{'przedzial':>12s} {'ile':>4s} " + " ".join(f"{g:>12s}" for g in names))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        if not m.any():
            continue
        shares = " ".join(f"{(group[m] == g).mean():12.0%}" for g in names)
        print(f"{lo:5.2f}-{hi:4.2f} m {m.sum():4d} {shares}")


table("Wedlug odleglosci celu od barku", radius, [0.2, 0.35, 0.5, 0.65, 0.8, 0.91])
table("Wedlug odleglosci na starcie", start, [0.0, 0.4, 0.8, 1.2, 1.6, 2.01])
