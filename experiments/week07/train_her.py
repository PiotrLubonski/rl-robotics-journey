"""Tydzien 7: SAC bez HER kontra SAC z HER (FetchReach albo wlasne PointReach)."""

# 1. NARZEDZIA
import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from point_reach import PointReach
from stable_baselines3 import SAC, HerReplayBuffer

gym.register_envs(gymnasium_robotics)
RUN = Path("runs/week07")

# 2. USTAWIENIA
parser = argparse.ArgumentParser()
parser.add_argument("--her", action="store_true", help="wlacz HER")
parser.add_argument(
    "--point", action="store_true", help="PointReach zamiast FetchReach"
)
parser.add_argument(
    "--threshold", type=float, default=0.05, help="prog trafienia w PointReach"
)
parser.add_argument("--env", default="FetchReach-v4")
parser.add_argument("--steps", type=int, default=20_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-episodes", type=int, default=50)
args = parser.parse_args()


def make_env():
    if args.point:
        return PointReach(threshold=args.threshold)
    return gym.make(args.env)


env_name = f"point{args.threshold}" if args.point else args.env
name = f"sac_her{int(args.her)}_{env_name}_{args.steps}_seed{args.seed}"
RESULTS = RUN / ("results_point.csv" if args.point else "results.csv")
RUN.mkdir(parents=True, exist_ok=True)

# 3. MODEL: jedyna roznica miedzy wariantami to bufor
extra = {}
if args.her:
    extra = dict(
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    )

model = SAC(
    "MultiInputPolicy",
    make_env(),
    seed=args.seed,
    verbose=1,
    device="cpu",
    gamma=0.95,
    learning_starts=1000,
    tensorboard_log=str(RUN / "tb"),
    **extra,
)

# 4. TRENING
start = time.perf_counter()
model.learn(total_timesteps=args.steps, tb_log_name=name)
minutes = (time.perf_counter() - start) / 60
model.save(RUN / name)

# 5. OCENA: 50 nowych celow, te same dla wszystkich wariantow
eval_env = make_env()
successes, dists = [], []
for ep in range(args.eval_episodes):
    obs, info = eval_env.reset(seed=1000 + ep)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            break
    successes.append(info["is_success"])
    dists.append(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

# 6. ZAPIS
row = {
    "her": int(args.her),
    "steps": args.steps,
    "seed": args.seed,
    "success_pct": round(100 * float(np.mean(successes))),
    "dist_cm": round(100 * float(np.mean(dists)), 1),
    "minutes": round(minutes, 1),
}
if args.point:
    row = {
        "threshold": args.threshold,
        **row,
    }  # PointReach: dodatkowa kolumna na poczatku
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(
    f"{name}: sukces {row['success_pct']}%, odleglosc {row['dist_cm']} "
    f"(x100 jednostek) ({row['minutes']} min)"
)
