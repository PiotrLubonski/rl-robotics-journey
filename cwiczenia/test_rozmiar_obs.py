import pytest

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import StackObs


def rozmiar_obs(angle_obs, with_velocity, n_klatek):
    """Ile liczb ma obserwacja ArmReach po sklejeniu n_klatek klatek."""
    if angle_obs == "cossin":
        katy = 4
    else:
        katy = 2
    # ↑ cossin: cos i sin dwóch kątów; raw: same dwa kąty (jedna linijka z if/else,
    #   jak n_angles w ArmReach.__init__ w arm_reach.py)
    if with_velocity:
        predkosc = 2
    else:
        predkosc = 0
    # ↑ ile liczb dokłada prędkość, gdy with_velocity jest True, a ile, gdy False
    cel_i_wektor = 4
    # ↑ cel (x, y) + wektor końcówka→cel (x, y)
    klatka = katy + predkosc + cel_i_wektor
    return n_klatek * klatka
    # ↑ StackObs skleja n_klatek CAŁYCH klatek


@pytest.mark.parametrize("angle_obs", ["cossin", "raw"])
@pytest.mark.parametrize("with_velocity", [True, False])
@pytest.mark.parametrize("n_klatek", [1, 2, 4])
def test_rozmiar_jak_w_srodowisku(angle_obs, with_velocity, n_klatek):
    env = StackObs(
        ArmReach(angle_obs=angle_obs, with_velocity=with_velocity), n=n_klatek
    )
    obs, _ = env.reset(seed=0)
    assert rozmiar_obs(angle_obs, with_velocity, n_klatek) == obs.shape[0]
