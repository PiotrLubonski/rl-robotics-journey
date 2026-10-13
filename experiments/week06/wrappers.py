"""Wrappery do HalfCheetah."""

import gymnasium as gym
import numpy as np


class FlipMonitor(gym.Wrapper):
    """Dopisuje do info, czy gepard sie przewrocil.

    Przewrotka = tulow obrocony o wiecej niz 90 stopni (cos(kat) < 0)
    przez co najmniej min_steps krokow z rzedu (20 krokow = 1 sekunda).
    """

    def __init__(self, env, min_steps=20):
        super().__init__(env)
        self.min_steps = min_steps
        self.upside_down = 0  # ile krokow z rzedu do gory nogami
        self.flipped = False  # czy w tym epizodzie byla przewrotka

    def reset(self, **kwargs):
        self.upside_down = 0
        self.flipped = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = self.unwrapped.data.qpos[2]  # obrot tulowia w radianach
        if np.cos(angle) < 0:
            self.upside_down += 1
        else:
            self.upside_down = 0
        if self.upside_down >= self.min_steps:
            self.flipped = True
        info["flipped"] = self.flipped
        return obs, reward, terminated, truncated, info


class UprightReward(gym.Wrapper):
    """Kara za przechylenie tulowia: weight * (cos(kat) - 1).

    0 gdy tulow poziomo, -weight przy 90 stopniach, -2 * weight na plecach.
    """

    def __init__(self, env, weight=1.0):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = self.unwrapped.data.qpos[2]
        penalty = self.weight * (np.cos(angle) - 1)
        info["upright_penalty"] = penalty
        return obs, reward + penalty, terminated, truncated, info
