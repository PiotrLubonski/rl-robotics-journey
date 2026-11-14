import numpy as np
import pytest

from rl_journey.envs.so101_reach import SO101Reach
from rl_journey.eval import evaluate_arm, evaluate_env
from rl_journey.wrappers import StackObs


class NicNieRobi:
    """Atrapa modelu: zawsze akcja zero."""

    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


class SprawdzaRozmiar:
    """Atrapa modelu: sprawdza, czy dostaje obserwację o oczekiwanym rozmiarze."""

    def __init__(self, size):
        self.size = size

    def predict(self, obs, deterministic=True):
        assert obs.shape == (self.size,)
        return np.zeros(2, dtype=np.float32), None


def test_nic_nie_robi_ma_zero_sukcesow():
    wynik = evaluate_arm(NicNieRobi(), episodes=20)
    assert wynik["success_pct"] == 0.0


def test_grupy_sumuja_sie_do_100():
    wynik = evaluate_arm(NicNieRobi(), episodes=20)
    suma = wynik["success_pct"] + wynik["not_closed_pct"] + wynik["missed_pct"]
    assert suma == pytest.approx(100.0, abs=0.2)


def test_ocena_przekazuje_opakowanie():
    # cossin bez prędkości = 8 liczb, dwie klatki = 16
    evaluate_arm(
        SprawdzaRozmiar(16),
        env_kwargs={"with_velocity": False},
        episodes=3,
        wrapper=lambda e: StackObs(e, n=2),
    )


class NicNieRobi5:
    """Atrapa modelu dla SO-101: zawsze akcja zero (5 stawów)."""

    def predict(self, obs, deterministic=True):
        return np.zeros(5, dtype=np.float32), None


def test_ocena_so101_nic_nie_robi():
    wynik = evaluate_env(NicNieRobi5(), SO101Reach(), episodes=5)
    assert wynik["success_pct"] == 0.0
    suma = wynik["success_pct"] + wynik["not_closed_pct"] + wynik["missed_pct"]
    assert suma == pytest.approx(100.0, abs=0.2)
