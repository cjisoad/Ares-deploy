from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.ares_mujoco_simulation import DEFAULT_BASE_QUAT, JOINT_ORDER, STAND_POSE


FOOT_SITE_NAMES = (
    "lf_calf_link_foot_site",
    "rf_calf_ink_foot_site",
    "lb_calf_link_foot_site",
    "rb_calf_link_foot_site",
)


@dataclass
class AresKinematics:
    model: mujoco.MjModel
    eps: float = 1.0e-4
    damping: float = 1.0e-4
    posture_weight: float = 0.25
    max_step: float = 0.12

    def __post_init__(self) -> None:
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER]
        self.qpos_addr = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.joint_ranges = self.model.jnt_range[self.joint_ids].astype(np.float32)
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES
        ]
        missing = [name for name, site_id in zip(FOOT_SITE_NAMES, self.foot_site_ids) if site_id < 0]
        if missing:
            raise ValueError(f"Missing foot sites in MJCF: {', '.join(missing)}")

    def foot_positions(self, joint_pos: np.ndarray, base_height: float = 0.8) -> np.ndarray:
        data = self._data_for_pose(joint_pos, base_height)
        base_pos = data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")]
        foot_positions = np.zeros((3, 4), dtype=np.float32)
        for leg_index, site_id in enumerate(self.foot_site_ids):
            foot_positions[:, leg_index] = data.site_xpos[site_id] - base_pos
        return foot_positions

    def inverse_kinematics(
        self,
        target_feet: np.ndarray,
        seed: np.ndarray,
        iterations: int = 4,
        base_height: float = 0.8,
    ) -> np.ndarray:
        q = np.asarray(seed, dtype=np.float32).reshape(12).copy()
        q_ref = q.copy()
        target = np.asarray(target_feet, dtype=np.float32).reshape(3, 4)
        posture_weight = np.diag(np.array([16.0, 1.0, 1.0] * 4, dtype=np.float32))
        for _ in range(iterations):
            current = self.foot_positions(q, base_height)
            error = (target - current).reshape(12)
            posture_error = q - q_ref
            if np.linalg.norm(error) < 1.0e-4 and np.linalg.norm(posture_error) < 1.0e-4:
                break
            jacobian = self._numeric_jacobian(q, base_height)
            lhs = (
                jacobian.T @ jacobian
                + self.damping * np.eye(12, dtype=np.float32)
                + self.posture_weight * posture_weight
            )
            rhs = jacobian.T @ error - self.posture_weight * posture_weight @ posture_error
            dq = np.linalg.solve(lhs, rhs)
            dq = np.clip(dq, -self.max_step, self.max_step)
            q = self.clip_joint_positions(q + dq)
        return q.astype(np.float32)

    def clip_joint_positions(self, joint_pos: np.ndarray) -> np.ndarray:
        return np.clip(joint_pos, self.joint_ranges[:, 0], self.joint_ranges[:, 1]).astype(np.float32)

    def _numeric_jacobian(self, joint_pos: np.ndarray, base_height: float) -> np.ndarray:
        base = self.foot_positions(joint_pos, base_height).reshape(12)
        jacobian = np.zeros((12, 12), dtype=np.float32)
        for index in range(12):
            q_step = joint_pos.copy()
            q_step[index] += self.eps
            moved = self.foot_positions(q_step, base_height).reshape(12)
            jacobian[:, index] = (moved - base) / self.eps
        return jacobian

    def _data_for_pose(self, joint_pos: np.ndarray, base_height: float) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        data.qpos[:] = 0.0
        data.qpos[2] = base_height
        data.qpos[3:7] = DEFAULT_BASE_QUAT
        data.qpos[self.qpos_addr] = np.asarray(joint_pos, dtype=np.float32).reshape(12)
        mujoco.mj_forward(self.model, data)
        return data


def default_foot_locations(model: mujoco.MjModel, base_height: float = 0.8) -> np.ndarray:
    return AresKinematics(model).foot_positions(STAND_POSE, base_height)
