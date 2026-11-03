"""Ocena polityki na ArmReach: sukces, grupy porażek, kraniec zasięgu."""

import numpy as np

from rl_journey.envs.arm_reach import ArmReach


def evaluate_arm(model, env_kwargs=None, episodes=200, seed=1000, edge=0.8):
    """Gra `episodes` epizodów i zwraca słownik z wynikami w procentach i cm."""
    env = ArmReach(**(env_kwargs or {}))
    end, ever, radius = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        radius.append(np.linalg.norm(env.target))  # jak daleko cel od barku
        hit = False
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            hit = hit or info["is_success"] == 1.0
        end.append(info["distance"])
        ever.append(hit)
    end, ever, radius = np.array(end), np.array(ever), np.array(radius)
    success = end < env.threshold
    at_edge = radius >= edge  # kraniec zasięgu (osobliwość)
    return {
        "episodes": episodes,
        "success_pct": float(round(100 * success.mean(), 1)),
        "not_closed_pct": float(round(100 * (ever & ~success).mean(), 1)),
        "missed_pct": float(round(100 * (~ever).mean(), 1)),
        "end_cm": float(round(100 * end.mean(), 1)),
        "edge_n": int(at_edge.sum()),
        "edge_success_pct": float(round(100 * success[at_edge].mean(), 1)),
    }
