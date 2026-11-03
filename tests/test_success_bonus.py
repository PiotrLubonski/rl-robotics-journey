import numpy as np
import pytest

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import SuccessBonus


def test_premia_gdy_koncowka_w_progu():
    env = SuccessBonus(ArmReach(), bonus=1.0)
    env.reset(seed=0)
    arm = env.unwrapped
    arm.target = arm.fingertip(arm.q)  # teleport celu dokładnie na końcówkę
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert info["is_success"] == 1.0
    # ↑ odległość = 0, próg 5 cm. W arm_reach.py w step(): "is_success": float(dist < self.threshold)
    assert reward == pytest.approx(1.0)
    # ↑ −odległość (0) − 0,01·akcja² (akcja zero) + premia z wrappera


def test_brak_premii_daleko_od_celu():
    env = SuccessBonus(ArmReach(), bonus=1.0)
    env.reset(seed=0)  # przy seed 0 start jest ok. 96 cm od celu
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert info["success_bonus"] == 0.0
    # ↑ ile premii daje wrapper, gdy is_success = 0? (wzór w SuccessBonus.step)
    assert reward == pytest.approx(-info["distance"])
