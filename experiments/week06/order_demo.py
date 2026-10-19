"""Pokaz: kto widzi jaka nagrode? Kolejnosc wrapperow."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from wrappers import UprightReward


class FakeData:
    def __init__(self):
        self.qpos = np.zeros(9)
        self.qpos[2] = np.pi  # gepard lezy na plecach


class BackCheetah(gym.Env):
    """Udawany gepard na plecach: 1.6 pkt za krok, epizod 1000 krokow."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,))

    def __init__(self):
        self.data = FakeData()
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(17), {}

    def step(self, action):
        self.t += 1
        return np.zeros(17), 1.6, False, self.t >= 1000, {}


def run(env, name):
    env.reset()
    seen_by_ppo = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(np.zeros(6))
        seen_by_ppo += reward  # to, co wychodzi z ostatniej nakladki
        if terminated or truncated:
            break
    print(
        f"{name}: Monitor zapisal {info['episode']['r']:6.0f}, "
        f"PPO dostaloby {seen_by_ppo:6.0f}"
    )


# Nasz uklad (tak robi make_vec_env): Monitor pierwszy, kara potem
run(UprightReward(Monitor(BackCheetah()), weight=1.0), "Monitor -> kara")
# Odwrotnie: kara pierwsza, Monitor potem
run(Monitor(UprightReward(BackCheetah(), weight=1.0)), "kara -> Monitor")
