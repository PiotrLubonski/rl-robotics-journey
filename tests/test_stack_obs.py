import numpy as np
import pytest

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import StackObs


def zrob(n=2, **kwargs):
    return StackObs(ArmReach(with_velocity=False, **kwargs), n=n)


def test_rozmiar_obserwacji():
    env = zrob(n=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (16,)
    # ↑ arm_reach.py, ArmReach.__init__: n_obs = n_angles + n_vel + 4
    #   (cossin: n_angles = 4; bez prędkości: n_vel = 0), a StackObs skleja n = 2 takie obserwacje
    assert env.observation_space.shape == obs.shape


def test_tasma_przesuwa_sie_o_jedna_klatke():
    env = zrob(n=2)
    obs0, _ = env.reset(seed=0)
    obs1, *_ = env.step(np.ones(2))  # *_ = reszta (nagroda, koniec, info), nieużywana
    assert np.array_equal(obs1[:8], obs0[8:])
    # ↑ wrappers.py, StackObs._stacked: klatki po 8 liczb, od najstarszej do najnowszej.
    #   Najnowsza klatka obs0 (jego druga połowa: od którego indeksu?) staje się najstarszą w obs1


def test_reset_czysci_pamiec():
    env = zrob(n=2)
    obs_start, _ = env.reset(seed=0)
    for _ in range(10):
        env.step(np.ones(2))
    obs_znowu, _ = env.reset(seed=0)
    assert np.array_equal(obs_znowu, obs_start)
    # ↑ ten sam seed = ten sam start. Z którą zmienną z TEGO testu ma być identyczna,
    #   jeśli reset wyczyścił taśmę? (tydzień 6: reset musi czyścić pamięć wrappera)


def test_z_dwoch_klatek_da_sie_policzyc_predkosc():
    env = zrob(n=2, angle_obs="raw")  # raw: klatka = 6 liczb, pierwsze 2 to q / π
    env.reset(seed=0)
    obs, *_ = env.step(np.array([1.0, 0.0]))
    q_stare = obs[0:2] * np.pi  # klatka 1
    q_nowe = obs[6:8] * np.pi  # klatka 2
    predkosc = (q_nowe - q_stare) / env.unwrapped.dt
    assert predkosc == pytest.approx(env.unwrapped.qd, abs=1e-4)
