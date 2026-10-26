from arm_reach import ArmReach
from gymnasium.utils.env_checker import check_env

for mode in ["cossin", "raw"]:
    for vel in [True, False]:
        check_env(ArmReach(angle_obs=mode, with_velocity=vel))
        print(f"{mode}, predkosc={vel}: OK")
