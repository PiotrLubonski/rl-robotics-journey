"""Rysunek: jak wytrenowane ramię jedzie do celu (kilka epizodów obok siebie)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rl_journey.envs.arm_reach import ArmReach
from stable_baselines3 import SAC

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--angle-obs", choices=["cossin", "raw"], default="cossin")
parser.add_argument("--no-velocity", action="store_true")
parser.add_argument("--episodes", type=int, default=3)
args = parser.parse_args()

model = SAC.load(args.model, device="cpu")
env = ArmReach(angle_obs=args.angle_obs, with_velocity=not args.no_velocity)
fig, axes = plt.subplots(
    2, args.episodes, figsize=(4 * args.episodes, 8), squeeze=False
)

for i in range(args.episodes):
    obs, info = env.reset(seed=1000 + i)
    qs = [env.q.copy()]
    dists = [np.linalg.norm(env.fingertip(env.q) - env.target)]
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        qs.append(env.q.copy())
        dists.append(info["distance"])

    # góra: ramię co 10 kroków, im później, tym ciemniej
    ax = axes[0, i]
    for k in range(0, len(qs), 10):
        q = qs[k]
        elbow = env.l1 * np.array([np.cos(q[0]), np.sin(q[0])])
        tip = env.fingertip(q)
        alpha = 0.15 + 0.85 * k / (len(qs) - 1)
        ax.plot(
            [0, elbow[0], tip[0]],
            [0, elbow[1], tip[1]],
            "-o",
            color="tab:blue",
            alpha=alpha,
        )
    tips = np.array([env.fingertip(q) for q in qs])
    ax.plot(tips[:, 0], tips[:, 1], ":", color="gray", label="droga końcówki")
    ax.plot(env.target[0], env.target[1], "r*", markersize=15, label="cel")
    ax.add_patch(plt.Circle(env.target, env.threshold, color="r", fill=False))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"epizod {i}: koniec {100 * dists[-1]:.1f} cm")
    ax.legend(loc="lower left", fontsize=8)

    # dół: odległość od celu w czasie
    ax = axes[1, i]
    ax.plot(100 * np.array(dists), label="odległość")
    ax.axhline(100 * env.threshold, color="r", linestyle="--", label="próg 5 cm")
    ax.set_xlabel("krok")
    ax.set_ylabel("cm do celu")
    ax.legend()

out = Path(args.model).with_suffix(".png")
fig.tight_layout()
fig.savefig(out, dpi=100)
print(f"zapisano {out}")
