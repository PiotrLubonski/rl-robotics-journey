"""Testy PointReach. Najwazniejsze: compute_reward dla paczki, bo tak wola go HER."""

import numpy as np
from point_reach import PointReach


def test_nagroda_dla_jednego_punktu():
    env = PointReach(threshold=0.05)
    assert (
        env.compute_reward(np.array([0.0, 0.0]), np.array([0.03, 0.0]), {}) == 0.0
    )  # 3 < 5
    assert (
        env.compute_reward(np.array([0.0, 0.0]), np.array([0.10, 0.0]), {}) == -1.0
    )  # 10 > 5


def test_nagroda_dla_paczki_jak_w_HER():
    env = PointReach(threshold=0.05)
    achieved = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    desired = np.array([[0.0, 0.01], [0.0, 0.5], [0.5, 0.5]])
    rewards = env.compute_reward(achieved, desired, {})
    assert rewards.shape == (3,)  # jedna nagroda na kazde przejscie
    assert list(rewards) == [0.0, -1.0, 0.0]


def test_ten_sam_seed_ten_sam_cel():
    env = PointReach()
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1["desired_goal"], obs2["desired_goal"])


def test_punkt_nie_wyjezdza_poza_plansze():
    env = PointReach()
    env.reset(seed=0)
    for _ in range(50):
        obs, _, _, _, _ = env.step(np.array([1.0, 1.0]))  # caly czas w prawy gorny rog
    assert np.all(obs["achieved_goal"] <= 1.0)


def test_epizod_konczy_sie_po_max_steps():
    env = PointReach(max_steps=50)
    env.reset(seed=0)
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert truncated is True and terminated is False
