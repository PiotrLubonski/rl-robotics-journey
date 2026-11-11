"""Tydzień 9: SAC na ArmReach z domain randomization (losowa fizyka + szum czujników)."""

import argparse
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import ObsNoise, RandomizePhysics

RUN = Path("runs/week09")

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--noise", type=float, default=0.02)
args = parser.parse_args()

name = f"sac_arm_dr_noise{args.noise:g}_{args.steps}_seed{args.seed}".replace(".", "p")
RUN.mkdir(parents=True, exist_ok=True)

env = Monitor(ObsNoise(RandomizePhysics(ArmReach()), std=args.noise))

model = SAC(
    "MlpPolicy",
    env,
    seed=args.seed,
    verbose=1,
    device="cpu",
    tensorboard_log=str(RUN / "tb"),
)
start = time.perf_counter()
model.learn(total_timesteps=args.steps, tb_log_name=name, log_interval=10)
minutes = (time.perf_counter() - start) / 60
model.save(RUN / name)
print(f"{name}: zapisany po {minutes:.1f} min. Ocena: robustness.py")
