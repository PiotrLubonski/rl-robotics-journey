"""Ocena polityki na ArmReach: sukces, grupy porażek, kraniec zasięgu."""

import numpy as np

from rl_journey.envs.arm_reach import ArmReach


def evaluate_arm(
    model, env_kwargs=None, episodes=200, seed=1000, edge=0.8, wrapper=None
):
    """Gra `episodes` epizodów i zwraca słownik z wynikami w procentach i cm.

    wrapper: funkcja opakowująca ramię, np. lambda e: StackObs(e, n=2); None = bez opakowania.
    """
    arm = ArmReach(**(env_kwargs or {}))
    env = wrapper(arm) if wrapper else arm
    end, ever, radius = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        radius.append(np.linalg.norm(arm.target))  # jak daleko cel od barku
        hit = False
        for _ in range(arm.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            hit = hit or info["is_success"] == 1.0
        end.append(info["distance"])
        ever.append(hit)
    end, ever, radius = np.array(end), np.array(ever), np.array(radius)
    success = end < arm.threshold
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


def evaluate_env(model, env, episodes=200, seed=1000):
    """Ocena na dowolnym środowisku „sięgnij celu” (info ma distance i is_success)."""
    threshold = env.unwrapped.threshold
    end, ever = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        hit, done = False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            hit = hit or info["is_success"] == 1.0
            done = terminated or truncated
        end.append(info["distance"])
        ever.append(hit)
    end, ever = np.array(end), np.array(ever)
    success = end < threshold
    return {
        "episodes": episodes,
        "success_pct": float(round(100 * success.mean(), 1)),
        "not_closed_pct": float(round(100 * (ever & ~success).mean(), 1)),
        "missed_pct": float(round(100 * (~ever).mean(), 1)),
        "end_cm": float(round(100 * end.mean(), 1)),
    }
