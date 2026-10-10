"""Tydzien 5: ktore ustawienie z Zoo daje najwiecej?
Domyslne PPO + (opcjonalnie) normalizacja + (opcjonalnie) mniejsza sigma."""

# 1. NARZEDZIA
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

RUN = Path("runs/week05_ablation")
RESULTS = RUN / "results.csv"

# 2. USTAWIENIA Z LINII KOMEND
parser = argparse.ArgumentParser()
parser.add_argument("--norm", action="store_true", help="wlacz VecNormalize")
parser.add_argument("--log-std", type=float, default=0.0, help="0 = domyslne, -2 = Zoo")
parser.add_argument("--env", default="HalfCheetah-v5")
parser.add_argument("--steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-episodes", type=int, default=50)
args = parser.parse_args()

RUN.mkdir(parents=True, exist_ok=True)
name = f"ppo_norm{int(args.norm)}_logstd{args.log_std}_seed{args.seed}"

# 3. SRODOWISKO TRENINGOWE
env = make_vec_env(args.env, n_envs=1, seed=args.seed)
if args.norm:
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 4. TRENING (wszystko domyslne poza log_std_init)
model = PPO(
    "MlpPolicy",
    env,
    seed=args.seed,
    verbose=0,
    device="cpu",
    tensorboard_log=str(RUN / "tb"),
    policy_kwargs=dict(log_std_init=args.log_std),
)
start = time.perf_counter()
model.learn(total_timesteps=args.steps, tb_log_name=name)
minutes = (time.perf_counter() - start) / 60
model.save(RUN / name)

# 5. EWALUACJA: 50 epizodow, liczymy przewrotki
# seed 0 -> starty 1000, te same co w eval_many
eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed + 1000)
if args.norm:
    stats = RUN / f"{name}_vecnormalize.pkl"
    env.save(str(stats))
    eval_env = VecNormalize.load(str(stats), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

rewards, _ = evaluate_policy(
    model, eval_env, n_eval_episodes=args.eval_episodes, return_episode_rewards=True
)
rewards = np.array(rewards)
fails = rewards < 1000  # przewrotka ~600 pkt
ok = rewards[~fails]
mean_ok = float(ok.mean()) if ok.size else 0.0

# 6. ZAPIS WYNIKU
row = {
    "norm": int(args.norm),
    "log_std": args.log_std,
    "seed": args.seed,
    "mean": round(float(rewards.mean()), 1),
    "fail_pct": round(100 * float(fails.mean())),
    "mean_ok": round(mean_ok, 1),
    "minutes": round(minutes, 1),
}
new_file = not RESULTS.exists()
with open(RESULTS, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        writer.writeheader()
    writer.writerow(row)
print(
    f"{name}: srednia {row['mean']}, porazki {row['fail_pct']}%, "
    f"udane {row['mean_ok']} ({row['minutes']} min)"
)
