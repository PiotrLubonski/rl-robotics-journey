import gymnasium as gym
import pytest


def test_gym_make_tworzy_ramie():
    env = gym.make("rl_journey:ArmReach-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)  # ile liczb widzi sieć w wersji domyślnej?
    assert env.spec.id == "ArmReach-v0"  # pod jakim id zarejestrowaliśmy środowisko?


@pytest.mark.parametrize(
    "angle_obs, with_velocity, size",
    [
        ("raw", True, 8),
        ("cossin", False, 8),  # z tabeli z tygodnia 8
    ],
)
def test_gym_make_przekazuje_ustawienia(angle_obs, with_velocity, size):
    env = gym.make(
        "rl_journey:ArmReach-v0", angle_obs=angle_obs, with_velocity=with_velocity
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (size,)
