import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_journey.envs.so101_reach import SO101Reach


def test_check_env():
    check_env(SO101Reach(), skip_render_check=True)


def test_rozmiar_obserwacji():
    obs, _ = SO101Reach().reset(seed=0)
    assert obs.shape == (24,)
    # ↑ ile liczb ma obserwacja? to Twoja H1 (policz z _obs w so101_reach.py)


def test_ten_sam_seed_ten_sam_start():
    obs_a, _ = SO101Reach().reset(seed=5)
    obs_b, _ = SO101Reach().reset(seed=5)
    assert np.array_equal(obs_a, obs_b)
    # ↑ zmienna z TEGO testu: ten sam seed ma dać tę samą obserwację startową


def test_kat_docelowy_nie_wychodzi_poza_zakres():
    env = SO101Reach()
    env.reset(seed=0)
    for _ in range(100):
        env.step(np.ones(5))  # cały czas „jeszcze dalej”
    target = env.data.ctrl[env.act_ids]
    assert np.all(target <= env.ctrl_hi + 1e-9)
    # ↑ so101_reach.py, __init__: zmienna z górną granicą sterowania 5 stawów (env.…)


def test_cel_nad_stolem_i_w_zasiegu():
    env = SO101Reach()
    for s in range(30):
        env.reset(seed=s)
        radius = np.linalg.norm(env.goal[:2] - env.base_xy)
        assert 0.02 <= env.goal[2] <= 0.30
        assert 0.12 <= radius <= 0.38


def test_nic_nie_robi_ramie_stoi():
    env = SO101Reach()
    env.reset(seed=0)
    start = env.tip()
    for _ in range(20):
        env.step(np.zeros(5))
    assert np.linalg.norm(env.tip() - start) < 0.01  # mniej niż 1 cm


def test_bez_akcji_nagroda_to_minus_odleglosc():
    env = SO101Reach()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5))
    assert reward == pytest.approx(-info["distance"])
