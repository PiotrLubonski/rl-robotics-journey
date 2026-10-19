import numpy as np
import pytest

from rl_core import ppo_updates, q_update, smooth_penalty, td_target


def test_cel_na_koncu_epizodu_to_sama_nagroda():
    # wpadl do dziury albo doszedl do celu: przyszlosci nie ma
    assert td_target(reward=1.0, next_q_max=5.0, gamma=0.99, terminated=True) == 1.0


def test_cel_w_srodku_epizodu_dodaje_przyszlosc():
    # 0 + 0.9 * 10 = 9
    assert td_target(
        reward=0.0, next_q_max=10.0, gamma=0.9, terminated=False
    ) == pytest.approx(9.0)


def test_alfa_zero_nic_nie_zmienia():
    assert q_update(q=3.0, target=100.0, alpha=0.0) == 3.0


def test_alfa_jeden_przepisuje_cel():
    assert q_update(q=3.0, target=100.0, alpha=1.0) == 100.0


def test_mieszanie_farb():
    # 0.9 * 0 + 0.1 * 5 = 0.5
    assert q_update(q=0.0, target=5.0, alpha=0.1) == pytest.approx(0.5)


def test_patrzymy_przyszlosc():
    assert td_target(reward=2.0, next_q_max=100.0, gamma=0.0, terminated=False) == 2.0


@pytest.mark.parametrize(
    "n_envs, n_steps, batch_size, n_epochs, expected",
    [
        (8, 1024, 64, 4, 512),  # Twoje PPO z LunarLandera
        (4, 256, 128, 5, 40),  # pytanie z dzisiejszego quizu
        (1, 512, 64, 20, 160),  # gepard z ustawieniami Zoo
    ],
)
def test_ppo_updates(n_envs, n_steps, batch_size, n_epochs, expected):
    assert ppo_updates(n_envs, n_steps, batch_size, n_epochs) == expected


def test_smooth_penalty_z_dzisiejszego_quizu():
    prev = np.array([0.3, 0, 0, 0, 0, 0])
    now = np.array([-0.3, 0.2, 0, 0, 0, 0])
    assert smooth_penalty(prev, now, weight=0.1) == pytest.approx(-0.04)


def test_smooth_penalty_skok_kosztuje_wiecej_niz_dwa_male_ruchy():
    prev = np.zeros(6)
    skok = np.array([1.0, 0, 0, 0, 0, 0])  # jeden staw o 1
    dwa_male = np.array([0.5, 0.5, 0, 0, 0, 0])  # dwa stawy po 0.5
    assert smooth_penalty(prev, skok, 0.1) < smooth_penalty(prev, dwa_male, 0.1)
