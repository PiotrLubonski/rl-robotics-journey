"""ArmReach: ramię z dwoma stawami na płaszczyźnie ma dotknąć celu końcówką."""

import gymnasium as gym
import numpy as np


def wrap_angle(q):
    """Zawija kąt do przedziału [-pi, pi). Działa dla liczby i dla tablicy."""
    return (q + np.pi) % (2 * np.pi) - np.pi


class ArmReach(gym.Env):
    def __init__(
        self,
        angle_obs="cossin",
        with_velocity=True,
        max_steps=100,
        threshold=0.05,
        link_lengths=(0.5, 0.5),
    ):
        self.angle_obs = angle_obs  # "cossin" albo "raw"
        self.with_velocity = with_velocity  # czy agent widzi prędkości stawów
        self.max_steps = max_steps
        self.threshold = threshold  # sukces: końcówka bliżej niż 5 cm
        self.l1, self.l2 = link_lengths  # długości ramienia i przedramienia (m)

        self.dt = 0.05  # 20 kroków na sekundę
        self.gain = 10.0  # jak mocno akcja rozpędza staw
        self.damping = 1.0  # tarcie: hamuje staw
        self.max_speed = 5.0  # limit silnika (rad/s)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_angles = 4 if angle_obs == "cossin" else 2
        n_vel = 2 if with_velocity else 0
        n_obs = n_angles + n_vel + 4  # + cel (x, y) + wektor końcówka→cel
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(n_obs,), dtype=np.float32
        )

    def fingertip(self, q):
        """Kinematyka prosta: kąty stawów → pozycja końcówki (x, y)."""
        elbow = self.l1 * np.array([np.cos(q[0]), np.sin(q[0])])
        a = q[0] + q[1]
        return elbow + self.l2 * np.array([np.cos(a), np.sin(a)])

    def _obs(self):
        if self.angle_obs == "cossin":
            angles = np.concatenate([np.cos(self.q), np.sin(self.q)])
        else:
            angles = self.q / np.pi  # od -1 do 1
        parts = [angles]
        if self.with_velocity:
            parts.append(self.qd / self.max_speed)  # od -1 do 1
        tip = self.fingertip(self.q)
        parts.append(self.target)
        parts.append(self.target - tip)  # dokąd jeszcze jechać
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q = self.np_random.uniform(-np.pi, np.pi, size=2)
        self.qd = np.zeros(2)
        angle = self.np_random.uniform(-np.pi, np.pi)
        radius = self.np_random.uniform(0.2, 0.9)
        self.target = radius * np.array([np.cos(angle), np.sin(angle)])
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        acc = self.gain * action - self.damping * self.qd
        self.qd = np.clip(self.qd + self.dt * acc, -self.max_speed, self.max_speed)
        self.q = wrap_angle(self.q + self.dt * self.qd)
        self.t += 1

        dist = float(np.linalg.norm(self.fingertip(self.q) - self.target))
        reward = -dist - 0.01 * float(np.sum(action**2))
        truncated = self.t >= self.max_steps
        info = {"distance": dist, "is_success": float(dist < self.threshold)}
        return self._obs(), reward, False, truncated, info


if __name__ == "__main__":
    env = ArmReach()
    env.action_space.seed(0)
    zero = np.zeros(2, dtype=np.float32)
    for name, policy in [
        ("nic nie robi", lambda: zero),
        ("losowy", env.action_space.sample),
    ]:
        start, end, ever = [], [], []
        for ep in range(200):
            obs, info = env.reset(seed=ep)
            start.append(np.linalg.norm(obs[-2:]))
            hit = False
            for _ in range(env.max_steps):
                obs, reward, terminated, truncated, info = env.step(policy())
                hit = hit or info["is_success"] == 1.0
            end.append(info["distance"])
            ever.append(hit)
        end = np.array(end)
        print(
            f"{name:13s} start {np.mean(start):.3f} m | koniec {end.mean():.3f} m | "
            f"sukces na koncu {(end < env.threshold).mean():.0%} | "
            f"dotknal choc raz {np.mean(ever):.0%}"
        )
