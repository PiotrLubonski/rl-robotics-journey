import numpy as np
import pytest

from rl_journey.eval import evaluate_arm


class NicNieRobi:
    """Atrapa modelu: zawsze akcja zero."""

    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_nic_nie_robi_ma_zero_sukcesow():
    wynik = evaluate_arm(NicNieRobi(), episodes=20)
    assert wynik["success_pct"] == 0.0


def test_grupy_sumuja_sie_do_100():
    wynik = evaluate_arm(NicNieRobi(), episodes=20)
    suma = wynik["success_pct"] + wynik["not_closed_pct"] + wynik["missed_pct"]
    assert suma == pytest.approx(100.0, abs=0.2)
