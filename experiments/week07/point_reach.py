"""Tydzien 7: wlasne mini-srodowisko z celami. Punkt ma dojechac do celu."""

import gymnasium as gym
import numpy as np


class PointReach(gym.Env):
    """Punkt na plaszczyznie [-1, 1] x [-1, 1] ma dojechac do losowego celu.

    Akcja: kierunek (dx, dy) z [-1, 1], przeskalowany do max 0.05 na krok.
    Nagroda rzadka: 0, gdy punkt jest blizej celu niz threshold, inaczej -1.
    """

    def __init__(self, threshold=0.05, max_steps=50, step_size=0.05):
        self.threshold = threshold
        self.max_steps = max_steps
        self.step_size = step_size
        box = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": box,
                "achieved_goal": box,
                "desired_goal": box,
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _obs(self):
        return {
            "observation": self.pos.copy(),  # kopie! (lekcja z b = a)
            "achieved_goal": self.pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2, dtype=np.float32)  # start na srodku
        self.goal = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        # dziala dla jednego punktu (2,) i dla paczki (n, 2): HER wola to z paczka!
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return -(dist > self.threshold).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.pos = np.clip(self.pos + self.step_size * action, -1, 1).astype(np.float32)
        self.t += 1
        reward = float(self.compute_reward(self.pos, self.goal, {}))
        truncated = self.t >= self.max_steps
        return (
            self._obs(),
            reward,
            False,
            truncated,
            {"is_success": float(reward == 0.0)},
        )


if __name__ == "__main__":
    # szybki sprawdzian: jak czesto trafia losowy agent przy dwoch progach?
    for threshold in (0.2, 0.05):
        env = PointReach(threshold=threshold)
        wins = 0
        for ep in range(200):
            obs, info = env.reset(seed=ep)
            while True:
                obs, r, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                if terminated or truncated:
                    break
            wins += info["is_success"]
        print(
            f"prog {threshold}: losowy agent trafia w {100 * wins / 200:.1f}% epizodow"
        )
