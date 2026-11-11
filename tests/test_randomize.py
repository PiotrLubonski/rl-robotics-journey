import numpy as np

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import ObsNoise, RandomizePhysics


def test_ten_sam_seed_ta_sama_fizyka():
    env = RandomizePhysics(ArmReach())
    env.reset(seed=3)
    l1_pierwszy = env.unwrapped.l1
    env.reset(seed=7)  # inny świat po drodze
    env.reset(seed=3)
    assert env.unwrapped.l1 == l1_pierwszy
    # ↑ zmienna z TEGO testu: ten sam seed ma dać ten sam świat
    #   (wrappers.py, RandomizePhysics.reset: przy podanym seedzie nowy self.rng)


def test_dlugosci_w_zakresie():
    env = RandomizePhysics(ArmReach())  # domyślnie link_scale=(0.9, 1.1)
    for s in range(50):
        env.reset(seed=s)
        assert 0.45 <= env.unwrapped.l1 <= 0.55
    # ↑ policz ręcznie: dolna granica to 0.5 · 0.9 = 0.45 (0.5 to l1 z link_lengths
    #   w ArmReach.__init__, arm_reach.py); górną policz tak samo z 1.1


def test_obserwacja_liczona_z_nowymi_dlugosciami():
    env = RandomizePhysics(ArmReach())
    obs, _ = env.reset(seed=0)
    arm = env.unwrapped
    assert np.allclose(obs[-2:], arm.target - arm.fingertip(arm.q), atol=1e-5)
    # ↑ arm_reach.py, _obs: ostatnie dwie liczby to wektor cel − końcówka.
    #   Wpisz wyrażenie z arm.target i arm.fingertip(arm.q), czyli z tego, co ramię ma TERAZ


def test_szum_zmienia_obserwacje_ale_nie_nagrode():
    czysty = ArmReach()
    zaszumiony = ObsNoise(ArmReach(), std=0.05)
    czysty.reset(seed=0)
    zaszumiony.reset(seed=0)
    obs_c, r_c, *_ = czysty.step(np.zeros(2))
    obs_z, r_z, *_ = zaszumiony.step(np.zeros(2))
    assert r_z == r_c  # nagroda liczona z prawdziwego stanu
    assert not np.array_equal(obs_z, obs_c)  # a czujniki widzą go z szumem
