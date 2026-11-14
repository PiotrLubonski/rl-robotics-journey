"""Tydzień 10: podgląd wytrenowanej polityki na SO101Reach w oknie MuJoCo."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

from rl_journey.envs.so101_reach import SO101Reach

parser = argparse.ArgumentParser()
parser.add_argument("model", help="ścieżka do modelu (bez .zip)")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--seed", type=int, default=1000)
args = parser.parse_args()

model = SAC.load(args.model, device="cpu")
env = SO101Reach()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        # cel jako czerwona, półprzezroczysta kulka o promieniu progu sukcesu
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([env.threshold, 0.0, 0.0]),
            pos=env.goal,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.1, 0.1, 0.5], dtype=np.float32),
        )
        for _ in range(env.max_steps):
            if not viewer.is_running():
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            viewer.sync()
            time.sleep(0.05)  # tempo rzeczywiste: 20 decyzji na sekundę
        wynik = "SUKCES" if info["is_success"] else "porażka"
        print(f"epizod {ep}: koniec {100 * info['distance']:.1f} cm od celu → {wynik}")
        time.sleep(1.0)
        if not viewer.is_running():
            break
