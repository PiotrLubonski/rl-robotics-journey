import numpy as np

from rl_journey.envs.arm_reach import ArmReach
from rl_journey.wrappers import ObsNoise, SuccessBonus

for std in [0.0, 0.1, 0.5]:
    env = ObsNoise(SuccessBonus(ArmReach(), bonus=2.0), std=std)
    env.reset(seed=0)
    obs, reward, *_ = env.step(np.array([0.5, 0.0]))
    print(
        f"std {std}: nagroda {reward:.4f} | pierwsze 3 liczby obserwacji {np.round(obs[:3], 3)}"
    )
