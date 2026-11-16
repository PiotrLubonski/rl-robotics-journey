import numpy as np

from rl_journey.baselines import JacobianController, SO101JacobianController
from rl_journey.envs.so101_reach import SO101Reach
from rl_journey.eval import evaluate_arm


def test_reczny_regulator_trafia():
    wynik = evaluate_arm(JacobianController(), episodes=20)
    assert wynik["success_pct"] == 100.0


def test_bez_ciagniecia_do_celu_nie_trafia():
    # kp = 0: regulator tylko hamuje, a ramię startuje z bezruchu, więc stoi jak „nic nie robi”
    wynik = evaluate_arm(JacobianController(kp=0.0), episodes=20)
    assert wynik["success_pct"] == 0.0


def test_regulator_so101_zbliza_sie_do_celu():
    env = SO101Reach()
    ctrl = SO101JacobianController()
    for s in range(3):
        obs, _ = env.reset(seed=s)
        start = np.linalg.norm(env.goal - env.tip())
        for _ in range(env.max_steps):
            action, _ = ctrl.predict(obs)
            obs, _, _, _, info = env.step(action)
        assert (
            info["distance"] < 0.5 * start
        )  # na końcu co najmniej 2 razy bliżej niż na starcie
