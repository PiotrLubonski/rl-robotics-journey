"""Punkty odniesienia bez uczenia: ręczny regulator dla ArmReach."""

import numpy as np


class JacobianController:
    """Ręczny regulator: pcha końcówkę w stronę celu (jakobian transponowany) i hamuje stawy.

    Zna tylko nominalne długości ramienia (0,5 m i 0,5 m) i działa na obserwacji
    cossin z prędkością (10 liczb), tak jak SAC. Ma metodę predict, więc evaluate_arm
    ocenia go tak samo jak wytrenowany model.
    """

    def __init__(self, kp=8.0, kd=0.18, l1=0.5, l2=0.5):
        self.kp = kp  # jak mocno ciągnąć końcówkę do celu
        self.kd = kd  # jak mocno hamować stawy
        self.l1, self.l2 = l1, l2

    def predict(self, obs, deterministic=True):
        c1, c2, s1, s2, v1, v2, tx, ty, dx, dy = obs
        q1 = np.arctan2(s1, c1)
        q12 = q1 + np.arctan2(s2, c2)
        jac = np.array(
            [
                [-self.l1 * np.sin(q1) - self.l2 * np.sin(q12), -self.l2 * np.sin(q12)],
                [self.l1 * np.cos(q1) + self.l2 * np.cos(q12), self.l2 * np.cos(q12)],
            ]
        )
        qd = 5.0 * np.array([v1, v2])  # obserwacja ma qd / 5
        action = self.kp * jac.T @ np.array([dx, dy]) - self.kd * qd
        return np.clip(action, -1.0, 1.0).astype(np.float32), None
