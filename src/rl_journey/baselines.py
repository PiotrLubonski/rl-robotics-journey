"""Punkty odniesienia bez uczenia: ręczny regulator dla ArmReach."""

import mujoco
import numpy as np

from rl_journey.envs.so101_reach import ARM_JOINTS, SCENE


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


class SO101JacobianController:
    """Ręczny regulator dla SO101Reach: odwrotna kinematyka krok po kroku z jakobianu MuJoCo.

    Z obserwacji bierze kąty stawów, kąty docelowe i wektor chwytak→cel. Liczy, jak zmienić
    kąty, żeby chwytak szedł prosto do celu, i zamienia to na akcję (zmianę kąta docelowego).
    """

    def __init__(self, max_delta=0.05, damping=0.02):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(
            self.model
        )  # własny „brudnopis”, nie rusza środowiska
        m = self.model
        joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS
        ]
        self.qpos_adr = m.jnt_qposadr[joint_ids]
        self.dof_adr = m.jnt_dofadr[joint_ids]
        self.tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.max_delta = max_delta  # jak w SO101Reach: 0,05 rad na krok
        self.damping = damping  # tłumienie przy osobliwości
        self.jacp = np.zeros((3, m.nv))

    def predict(self, obs, deterministic=True):
        q, target, error = (
            obs[0:5],
            obs[10:15],
            obs[21:24],
        )  # kąty, kąty docelowe, chwytak→cel
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.tip_id)
        jac = self.jacp[:, self.dof_adr]  # tabela 3 × 5
        lam2 = self.damping**2
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(3), error)
        action = (q + dq - target) / self.max_delta  # nowy kąt docelowy → akcja
        action = action / max(
            1.0, float(np.max(np.abs(action)))
        )  # skróć, zachowaj kierunek
        return action.astype(np.float32), None
