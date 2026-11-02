import numpy as np
import pytest
from rl_journey.envs.arm_reach import ArmReach, wrap_angle


@pytest.mark.parametrize(
    "q, expected",
    [
        ([0.0, 0.0], [1.0, 0.0]),  # wyprostowane w prawo
        ([np.pi / 2, 0.0], [0.0, 1.0]),  # wyprostowane w górę
        ([0.0, np.pi / 2], [0.5, 0.5]),  # łokieć zgięty o 90°
        ([0.0, np.pi], [0.0, 0.0]),  # przedramię złożone do tyłu: końcówka w barku
    ],
)
def test_koncowka(q, expected):
    env = ArmReach()
    tip = env.fingertip(np.array(q))
    assert tip == pytest.approx(np.array(expected), abs=1e-9)


@pytest.mark.parametrize(
    "angle_obs, with_velocity, size",
    [
        ("cossin", True, 10),
        ("cossin", False, 8),
        ("raw", True, 8),
        ("raw", False, 6),
    ],
)
def test_rozmiar_obserwacji(angle_obs, with_velocity, size):
    env = ArmReach(angle_obs=angle_obs, with_velocity=with_velocity)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (size,)
    assert env.observation_space.contains(obs)


def test_zawijanie_kata():
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert wrap_angle(0.3) == pytest.approx(0.3)


def test_ten_sam_seed_ten_sam_start():
    obs_a, _ = ArmReach().reset(seed=7)
    obs_b, _ = ArmReach().reset(seed=7)
    assert np.array_equal(obs_a, obs_b)


def test_predkosc_ma_limit():
    env = ArmReach()
    env.reset(seed=0)
    for _ in range(200):
        env.step(np.array([1.0, -1.0]))
    assert np.all(np.abs(env.qd) <= env.max_speed)


def test_koniec_po_max_steps():
    env = ArmReach(max_steps=5)
    env.reset(seed=0)
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert not truncated
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert truncated


def test_bez_akcji_nagroda_to_minus_odleglosc():
    env = ArmReach()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert reward == pytest.approx(-info["distance"])


# ---------- Twoja kolej ----------


def roznica_obserwacji(env, deg_a, deg_b):
    """Jak daleko od siebie są obserwacje dla dwóch kątów barku (łokieć prosty)?"""
    env.q = np.array([np.deg2rad(deg_a), 0.0])  # teleport do pozy A
    obs_a = env._obs()  # co widzi sieć w pozie A
    env.q = np.array([np.deg2rad(deg_b), 0.0])  # teleport do pozy B
    obs_b = env._obs()  # co widzi sieć w pozie B
    return np.linalg.norm(obs_a - obs_b)  # jak daleko od siebie


def test_cossin_widzi_sasiadow_przy_180_stopniach():
    env_cs = ArmReach(angle_obs="cossin", with_velocity=False)
    env_raw = ArmReach(angle_obs="raw", with_velocity=False)
    env_cs.reset(seed=0)
    env_raw.reset(seed=0)
    d_cs = roznica_obserwacji(env_cs, 179.0, -179.0)
    d_raw = roznica_obserwacji(env_raw, 179.0, -179.0)
    print(f"cossin: {d_cs:.3f}, raw: {d_raw:.3f}")
    assert d_cs < 0.1  # sąsiedzi mają być blisko
    assert d_raw > 1.0  # surowy kąt robi „dziurę” przy 180°
