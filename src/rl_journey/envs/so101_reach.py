"""SO101Reach: model ramienia SO-101 w MuJoCo ma dotknąć chwytakiem celu w 3D."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "assets" / "so101" / "scene.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class SO101Reach(gym.Env):
    """Akcja: zmiana kąta docelowego 5 stawów (serwa). Obserwacja: kąty, prędkości,
    kąty docelowe, chwytak, cel, wektor chwytak→cel."""

    def __init__(
        self,
        max_steps=100,
        threshold=0.02,
        max_delta=0.05,
        control_dt=0.05,
        start_noise=0.2,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.fk_data = mujoco.MjData(self.model)  # osobny „brudnopis” do losowania celu
        self.max_steps = max_steps
        self.threshold = threshold  # sukces: chwytak bliżej niż 2 cm
        self.max_delta = max_delta  # najwięcej rad zmiany kąta docelowego na krok
        self.start_noise = start_noise  # start: poza zerowa ± 0,2 rad
        self.n_substeps = max(1, round(control_dt / self.model.opt.timestep))

        m = self.model
        self.joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS
        ]
        self.act_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS
        ]
        self.qpos_adr = m.jnt_qposadr[self.joint_ids]
        self.dof_adr = m.jnt_dofadr[self.joint_ids]
        self.ctrl_lo = m.actuator_ctrlrange[self.act_ids, 0]
        self.ctrl_hi = m.actuator_ctrlrange[self.act_ids, 1]
        self.tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        mujoco.mj_kinematics(m, self.fk_data)
        self.base_xy = self.fk_data.xanchor[self.joint_ids[0]][:2].copy()  # oś podstawy

        n = len(ARM_JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        n_obs = 3 * n + 9
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(n_obs,), dtype=np.float32
        )

    def tip(self):
        """Położenie chwytaka (x, y, z) w metrach."""
        return self.data.site_xpos[self.tip_id].copy()

    def _sample_goal(self):
        """Losowa osiągalna poza → położenie jej chwytaka staje się celem."""
        lo = self.model.jnt_range[self.joint_ids, 0]
        hi = self.model.jnt_range[self.joint_ids, 1]
        for _ in range(1000):
            self.fk_data.qpos[self.qpos_adr] = self.np_random.uniform(
                0.8 * lo, 0.8 * hi
            )
            mujoco.mj_kinematics(self.model, self.fk_data)
            p = self.fk_data.site_xpos[self.tip_id]
            radius = np.linalg.norm(p[:2] - self.base_xy)
            if 0.02 <= p[2] <= 0.30 and 0.12 <= radius <= 0.38:
                return p.copy()
        raise RuntimeError("nie udało się wylosować celu")

    def _obs(self):
        q = self.data.qpos[self.qpos_adr]
        qd = self.data.qvel[self.dof_adr] / 5.0
        target = self.data.ctrl[self.act_ids]
        tip = self.tip()
        return np.concatenate([q, qd, target, tip, self.goal, self.goal - tip]).astype(
            np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        q0 = self.np_random.uniform(
            -self.start_noise, self.start_noise, size=len(ARM_JOINTS)
        )
        q0 = np.clip(q0, self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qpos_adr] = q0
        self.data.ctrl[self.act_ids] = q0  # serwa trzymają pozę startową
        mujoco.mj_forward(self.model, self.data)
        self.goal = self._sample_goal()
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = self.data.ctrl[self.act_ids] + self.max_delta * action
        self.data.ctrl[self.act_ids] = np.clip(target, self.ctrl_lo, self.ctrl_hi)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.t += 1

        dist = float(np.linalg.norm(self.tip() - self.goal))
        reward = -dist - 0.01 * float(np.sum(action**2))
        truncated = self.t >= self.max_steps
        info = {"distance": dist, "is_success": float(dist < self.threshold)}
        return self._obs(), reward, False, truncated, info
