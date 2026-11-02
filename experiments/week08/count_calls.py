"""Pokaz: kogo woła model.predict, a kogo trening (model.learn)."""

import argparse

from rl_journey.envs.arm_reach import ArmReach
from stable_baselines3 import SAC

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

model = SAC.load(args.model, env=ArmReach(), device="cpu")
model.tensorboard_log = None  # pokaz nie zaśmieca TensorBoarda

calls = {"aktor": 0, "krytyk": 0, "krytyk_docelowy": 0}


def licznik(nazwa):
    """Fabryka dzwonków: zwraca funkcję, która dolicza 1 przy każdym użyciu sieci."""

    def dzwonek(module, inputs, output):
        calls[nazwa] += 1

    return dzwonek


model.actor.mu.register_forward_hook(licznik("aktor"))
model.critic.register_forward_hook(licznik("krytyk"))
model.critic_target.register_forward_hook(licznik("krytyk_docelowy"))

# Część 1: robot w pracy (tak jak na prawdziwym ramieniu)
env = ArmReach()
obs, info = env.reset(seed=0)
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
print(f"predict, 100 krokow: {calls}")

# Część 2: trening (dalsza nauka przez 200 kroków)
for k in calls:
    calls[k] = 0
model.learn(total_timesteps=200, reset_num_timesteps=False)
print(f"learn, 200 krokow:   {calls}")
