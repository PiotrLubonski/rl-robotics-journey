"""PPO na LunarLander-v3 ze Stable-Baselines3."""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

RUN = Path("runs/week03_ppo_lunar")
TOTAL_STEPS = 500_000


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    train_env = make_vec_env("LunarLander-v3", n_envs=8, seed=0)

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=0,
        verbose=1,
        tensorboard_log=str(RUN / "tb"),
        device="cpu",
    )
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
    model.save(RUN / "ppo_lunar")

    eval_env = gym.make("LunarLander-v3")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"\nSredni zwrot (ewaluacja): {mean:.1f} +/- {std:.1f}")
    eval_env.close()


if __name__ == "__main__":
    main()
