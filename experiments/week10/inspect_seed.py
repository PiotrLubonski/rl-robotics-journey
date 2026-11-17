"""Tydzień 10: dlaczego regulator utknął? Kąty: start, poza celu, koniec, granice zakresu."""

import argparse

import numpy as np

from rl_journey.baselines import SO101JacobianController
from rl_journey.envs.so101_reach import ARM_JOINTS, SO101Reach

parser = argparse.ArgumentParser()
parser.add_argument("seeds", nargs="*", type=int, default=[1014, 1130, 1144])
args = parser.parse_args()

env, ctrl = SO101Reach(), SO101JacobianController()
deg = np.rad2deg
for seed in args.seeds:
    obs, _ = env.reset(seed=seed)
    start_q = env.data.qpos[env.qpos_adr].copy()
    for _ in range(env.max_steps):
        action, _ = ctrl.predict(obs)
        obs, _, _, _, info = env.step(action)
    end_q = env.data.qpos[env.qpos_adr].copy()

    print(f"\nseed {seed}: koniec {100 * info['distance']:.1f} cm od celu")
    print(
        f"  {'staw':14s} {'start':>7s} {'poza celu':>10s} {'koniec':>7s} {'zakres':>17s}  uwagi"
    )
    for i, joint in enumerate(ARM_JOINTS):
        lo, hi = env.ctrl_lo[i], env.ctrl_hi[i]
        notes = []
        if min(end_q[i] - lo, hi - end_q[i]) < np.deg2rad(3):
            notes.append("KONIEC PRZY GRANICY")
        if np.sign(env.goal_q[i]) != np.sign(end_q[i]) and abs(
            env.goal_q[i] - end_q[i]
        ) > np.deg2rad(30):
            notes.append("cel po drugiej stronie")
        print(
            f"  {joint:14s} {deg(start_q[i]):6.1f}° {deg(env.goal_q[i]):9.1f}° {deg(end_q[i]):6.1f}° "
            f"{deg(lo):7.1f}° … {deg(hi):5.1f}°  {', '.join(notes)}"
        )
