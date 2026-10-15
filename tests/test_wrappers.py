"""Testy wrapperow z tygodnia 6.

Zamiast prawdziwego geparda uzywamy atrapy: malego srodowiska,
w ktorym sami ustawiamy kat tulowia. Testy sa szybkie i przewidywalne.
"""

import gymnasium as gym
import numpy as np
import pytest
from wrappers import FlipMonitor, SmoothActionReward, UprightReward

# ---------- ATRAPA ----------


class FakeData:
    def __init__(self):
        self.qpos = np.zeros(9)  # jak w HalfCheetah: qpos[2] = kat tulowia


class FakeCheetah(gym.Env):
    """Udawany gepard: zawsze nagroda 1.0, kat tulowia ustawiamy recznie."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,))
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,))

    def __init__(self):
        self.data = FakeData()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(17), {}

    def step(self, action):
        return np.zeros(17), 1.0, False, False, {}


def set_angle(env, degrees):
    env.unwrapped.data.qpos[2] = np.radians(degrees)


ZERO = np.zeros(6)


# ---------- FlipMonitor ----------


def test_flip_dopiero_po_20_krokach_na_plecach():
    env = FlipMonitor(FakeCheetah(), min_steps=20)
    env.reset()
    set_angle(env, 180)
    for _ in range(19):
        _, _, _, _, info = env.step(ZERO)
    assert info["flipped"] is False  # 19 krokow to jeszcze za malo
    _, _, _, _, info = env.step(ZERO)
    assert info["flipped"] is True  # 20. krok: przewrotka


def test_flip_krotki_przechyl_sie_nie_liczy():
    env = FlipMonitor(FakeCheetah(), min_steps=20)
    env.reset()
    set_angle(env, 180)
    for _ in range(10):
        env.step(ZERO)  # 10 krokow do gory nogami...
    set_angle(env, 0)
    for _ in range(30):
        _, _, _, _, info = env.step(ZERO)  # ...i z powrotem na nogach
    assert info["flipped"] is False


def test_flip_reset_kasuje_przewrotke():
    env = FlipMonitor(FakeCheetah(), min_steps=20)
    env.reset()
    set_angle(env, 180)
    for _ in range(25):
        env.step(ZERO)
    env.reset()
    set_angle(env, 0)
    _, _, _, _, info = env.step(ZERO)
    assert info["flipped"] is False


def test_flip_pelny_obrot_to_nie_przewrotka():
    env = FlipMonitor(FakeCheetah(), min_steps=20)
    env.reset()
    set_angle(env, 360)  # obrot o 360 st. = znowu na nogach
    for _ in range(25):
        _, _, _, _, info = env.step(ZERO)
    assert info["flipped"] is False


def test_flip_obrot_minus_180():
    env = FlipMonitor(FakeCheetah(), min_steps=20)
    env.reset()
    set_angle(env, -180)  # sprawdzenie -180
    for _ in range(25):
        _, _, _, _, info = env.step(ZERO)
    assert info["flipped"] is True


# ---------- UprightReward ----------


@pytest.mark.parametrize(
    "degrees, expected",
    [
        (0, 1.0),  # poziomo: brak kary, zostaje 1.0
        (90, 0.0),  # pionowo: kara -1
        (180, -1.0),  # na plecach: kara -2
    ],
)
def test_upright_kara(degrees, expected):
    env = UprightReward(FakeCheetah(), weight=1.0)
    env.reset()
    set_angle(env, degrees)
    _, reward, _, _, _ = env.step(ZERO)
    assert reward == pytest.approx(expected)


# ---------- SmoothActionReward ----------


def test_smooth_pierwszy_krok_bez_kary():
    env = SmoothActionReward(FakeCheetah(), weight=0.1)
    env.reset()
    _, reward, _, _, _ = env.step(np.ones(6))
    assert reward == pytest.approx(1.0)


def test_smooth_kara_z_quizu():
    env = SmoothActionReward(FakeCheetah(), weight=0.1)
    env.reset()
    env.step(np.array([0.5, -1, 0, 0, 0, 0]))
    _, reward, _, _, _ = env.step(np.array([1.0, 1, 0, 0, 0, 0]))
    # zmiany 0.5 i 2 → 0.25 + 4 = 4.25 → razy 0.1 → kara -0.425
    assert reward == pytest.approx(1.0 - 0.425)


def test_smooth_reset_zapomina_poprzednia_akcje():
    env = SmoothActionReward(FakeCheetah(), weight=0.1)
    env.reset()
    env.step(-np.ones(6))
    env.reset()
    _, reward, _, _, _ = env.step(np.ones(6))  # po resecie nie ma z czym porownac
    assert reward == pytest.approx(1.0)
