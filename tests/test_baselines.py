from rl_journey.baselines import JacobianController
from rl_journey.eval import evaluate_arm


def test_reczny_regulator_trafia():
    wynik = evaluate_arm(JacobianController(), episodes=20)
    assert wynik["success_pct"] == 100.0


def test_bez_ciagniecia_do_celu_nie_trafia():
    # kp = 0: regulator tylko hamuje, a ramię startuje z bezruchu, więc stoi jak „nic nie robi”
    wynik = evaluate_arm(JacobianController(kp=0.0), episodes=20)
    assert wynik["success_pct"] == 0.0
