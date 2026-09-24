# RL Robotics Journey

My 3-month self-study of reinforcement learning for robotics — from tabular Q-learning to PPO, heading toward robot manipulation in MuJoCo and a real low-cost arm.

![PPO agent landing on the Moon](media/lunar_ppo.gif)

## Progress

| Week | Topic | Result |
|------|-------|--------|
| 1 | Random agent on CartPole | baseline: 21 steps |
| 2 | Tabular Q-learning from scratch on FrozenLake | 100% (deterministic), 74% (slippery — near the theoretical ceiling) |
| 3 | PPO on LunarLander (Stable-Baselines3) | **212.4 ± 74.0** — solved (threshold 200) |

## Experiment: how the discount factor γ changes behaviour

PPO on LunarLander-v3, 300k steps (shorter than the main run), 3 seeds per setting. Hypothesis written before running.

| γ | Horizon 1/(1−γ) | Mean return | Min | Max | Behaviour |
|---|---|---|---|---|---|
| 0.999 | 1000 | 102.4 | −42.1 | 192.2 | lands (~650 steps) |
| 0.99 | 100 | 5.2 | −128.6 | 183.3 | unstable — depends on the seed |
| 0.9 | 10 | −136.0 | −167.5 | −103.8 | hovers until the time limit |

![Short-sighted agent hovering](media/lunar_gamma09.gif)

**Takeaway.** The higher γ, the more it pays off for the agent to land; with low γ it prefers to hover and never risks a crash. The fuel burned while hovering is spread over hundreds of steps, so with γ = 0.9 the agent perceives the cost of hovering *forever* as only ~1.7 points — while a crash (−100) is right in front of it.

The difference between γ = 0.999 and γ = 0.99 is within seed-to-seed variance; only γ = 0.9 is clearly worse.

## Run it

```bash
uv sync
uv run python experiments/week03/ppo_lunar.py
uv run python experiments/week03/watch.py runs/week03_ppo_lunar/ppo_lunar
```

## Repository layout

```
experiments/   one-off scripts, grouped by week
notes/         notes and analogies (in Polish)
src/           reusable code (custom environment coming in week 8)
media/         GIFs for this README
JOURNAL.md     learning log (in Polish)
```