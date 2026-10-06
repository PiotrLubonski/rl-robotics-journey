"""PPO z ustawieniami z RL Baselines3 Zoo, z normalizacja."""

import argparse
import csv
import time
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

RUN = Path("runs/week05")
RESULTS = RUN / "results_zoo.csv"

# Zrodlo: rl-baselines3-zoo, hyperparams/ppo.yml, HalfCheetah-v4
ZOO_PPO = dict(
    batch_size=64,
    n_steps=512,
    gamma=0.98,
    learning_rate=2.0633e-05,
    ent_coef=0.000401762,
    clip_range=0.1,
    n_epochs=20,
    gae_lambda=0.92,
    max_grad_norm=0.8,
    vf_coef=0.58096,
    policy_kwargs=dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def save_result(row):
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    name = f"ppo_zoo_{args.steps}_seed{args.seed}_env{args.env}"
    stats_path = RUN / f"{name}_vecnormalize.pkl"

    # GDZIE — srodowisko, owiniete normalizacja
    venv = make_vec_env(args.env, n_envs=1, seed=args.seed)
    train_env = VecNormalize(venv, norm_obs=True, norm_reward=True)

    # CZYM — PPO z ustawieniami z Zoo
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=0,
        tensorboard_log=str(RUN / "tb"),
        device="cpu",
        **ZOO_PPO,
    )

    # JAK DLUGO
    start = time.perf_counter()
    model.learn(total_timesteps=args.steps, tb_log_name=name, progress_bar=True)
    minutes = (time.perf_counter() - start) / 60

    # CO ZACHOWAC — model ORAZ statystyki normalizacji
    model.save(RUN / name)
    train_env.save(str(stats_path))

    # JAK WYSZLO — te same statystyki, ale zamrozone i bez normalizacji nagrody
    eval_env = VecNormalize.load(
        str(stats_path), make_vec_env(args.env, n_envs=1, seed=args.seed + 100)
    )
    eval_env.training = False
    eval_env.norm_reward = False
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    eval_env.close()

    save_result(
        {
            "algo": "ppo_zoo",
            "env": args.env,
            "steps": args.steps,
            "seed": args.seed,
            "mean": round(mean, 1),
            "std": round(std, 1),
            "minutes": round(minutes, 1),
        }
    )
    print(f"{name}: {mean:.1f} +/- {std:.1f}  ({minutes:.1f} min)")


if __name__ == "__main__":
    main()
