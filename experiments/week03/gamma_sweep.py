"""Eksperyment: wplyw gamma na PPO w LunarLander-v3."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

RUN = Path("runs/week03_gamma")
RESULTS = RUN / "results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="PPO na LunarLander z wybranym gamma")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000)
    return parser.parse_args()


def save_result(gamma, seed, mean, std):
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["gamma", "seed", "mean", "std"])
        writer.writerow([gamma, seed, round(mean, 1), round(std, 1)])


def main():
    args = parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    name = f"gamma{args.gamma}_seed{args.seed}"

    train_env = make_vec_env("LunarLander-v3", n_envs=8, seed=args.seed)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=args.gamma,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=args.seed,
        verbose=0,
        tensorboard_log=str(RUN / "tb"),
        device="cpu",
    )
    model.learn(total_timesteps=args.steps, tb_log_name=name, progress_bar=True)
    model.save(RUN / name)

    eval_env = gym.make("LunarLander-v3")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()

    save_result(args.gamma, args.seed, mean, std)
    print(f"{name}: {mean:.1f} +/- {std:.1f}")


if __name__ == "__main__":
    main()
