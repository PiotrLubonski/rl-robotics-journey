import pytest

from rl_core import q_update, td_target


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
