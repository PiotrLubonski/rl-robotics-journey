"""Wrappery do HalfCheetah."""

from collections import deque

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
        self.upside_down = 0
        self.flipped = False

    def reset(self, **kwargs):
        self.upside_down = 0
        self.flipped = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = self.unwrapped.data.qpos[2]
        if np.cos(angle) < 0:
            self.upside_down += 1
        else:
            self.upside_down = 0
        if self.upside_down >= self.min_steps:
            self.flipped = True
        info["flipped"] = self.flipped
        return obs, reward, terminated, truncated, info


class UprightReward(gym.Wrapper):
    """Kara za przechylenie tulowia: weight * (cos(kat) - 1)."""

    def __init__(self, env, weight=1.0):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = self.unwrapped.data.qpos[2]
        penalty = self.weight * (np.cos(angle) - 1)
        info["upright_penalty"] = penalty
        return obs, reward + penalty, terminated, truncated, info


class SmoothActionReward(gym.Wrapper):
    """Kara za szarpanie: -weight * suma((a_t - a_{t-1})^2) po stawach."""

    def __init__(self, env, weight=0.1):
        super().__init__(env)
        self.weight = weight
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None  # nowy epizod: brak poprzedniej akcji
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.clip(action, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.prev_action is None:
            penalty = 0.0  # pierwszy krok: nie ma z czym porownac
        else:
            penalty = -self.weight * np.sum((action - self.prev_action) ** 2)
        self.prev_action = action.copy()  # kopia, bo SB3 moze nadpisac te sama tablice
        info["smooth_penalty"] = penalty
        return obs, reward + penalty, terminated, truncated, info


def make_shaped(env, upright=1.0, smooth=0.0):
    """Naklada obie kary naraz (make_vec_env przyjmuje jedna funkcje)."""
    env = UprightReward(env, weight=upright)
    env = SmoothActionReward(env, weight=smooth)
    return env


class SuccessBonus(gym.Wrapper):
    """Dodaje premię w każdym kroku, w którym końcówka jest w progu."""

    def __init__(self, env, bonus=1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self.bonus * info.get("is_success", 0.0)
        info["success_bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info


class StackObs(gym.Wrapper):
    """Obserwacja = n ostatnich obserwacji sklejonych w jedną (najstarsza pierwsza)."""

    def __init__(self, env, n=2):
        super().__init__(env)
        self.n = n
        size = env.observation_space.shape[0] * n
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        )
        self.frames = deque(
            maxlen=n
        )  # taśma na n klatek: nowa wchodzi, najstarsza wypada

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):  # na starcie cała taśma = pierwsza klatka
            self.frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self):
        return np.concatenate(self.frames).astype(np.float32)
