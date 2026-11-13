"""rl_journey: środowiska, wrappery i narzędzia z kursu RL w robotyce."""

from gymnasium.envs.registration import register

register(id="ArmReach-v0", entry_point="rl_journey.envs.arm_reach:ArmReach")
register(id="PointReach-v0", entry_point="rl_journey.envs.point_reach:PointReach")
register(id="SO101Reach-v0", entry_point="rl_journey.envs.so101_reach:SO101Reach")
