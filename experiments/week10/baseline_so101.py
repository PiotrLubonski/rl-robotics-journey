"""Tydzień 10: punkt zero na SO101Reach: „nic nie robi” i losowy agent (200 epizodów)."""

import numpy as np

from rl_journey.envs.so101_reach import SO101Reach

env = SO101Reach()
print(
    f"krok fizyki {env.model.opt.timestep * 1000:.1f} ms, {env.n_substeps} kroków fizyki na decyzję"
)
env.action_space.seed(0)
zero = np.zeros(5, dtype=np.float32)
for name, policy in [
    ("nic nie robi", lambda: zero),
    ("losowy", env.action_space.sample),
]:
    start, end, ever = [], [], []
    for ep in range(200):
        env.reset(seed=1000 + ep)
        start.append(np.linalg.norm(env.goal - env.tip()))
        hit = False
        for _ in range(env.max_steps):
            _, _, _, _, info = env.step(policy())
            hit = hit or info["is_success"] == 1.0
        end.append(info["distance"])
        ever.append(hit)
    end = np.array(end)
    print(
        f"{name:13s} start {100 * np.mean(start):5.1f} cm | koniec {100 * end.mean():5.1f} cm | "
        f"sukces na końcu {(end < env.threshold).mean():.0%} | dotknął choć raz {np.mean(ever):.0%}"
    )
