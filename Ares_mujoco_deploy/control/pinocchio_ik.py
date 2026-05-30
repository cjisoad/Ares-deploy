from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


PINOCCHIO_MESH_ROOT = Path("/home/eden/Ares_deploy/model")
URDF_PATH = Path("/home/eden/Ares_deploy/model/Ares/urdf/Ares.urdf")

PINOCCHIO_JOINT_ORDER = (
    "lb_hip_base_joint",
    "lb_thigh_hip_joint",
    "lb_calf_thigh_joint",
    "lf_hip_base_joint",
    "lf_thigh_hip_joint",
    "lf_calf_thigh_joint",
    "rb_hip_base_joint",
    "rb_thigh_hip_joint",
    "rb_calf_thigh_joint",
    "rf_hip_base_joint",
    "rf_thigh_hip_joint",
    "rf_calf_thigh_joint",
)

PINOCCHIO_FOOT_FRAMES = (
    "lf_calf_link",
    "rf_calf_ink",
    "lb_calf_link",
    "rb_calf_link",
)

FOOT_FRAME_NAMES = (
    "lf_foot",
    "rf_foot",
    "lb_foot",
    "rb_foot",
)

FOOT_OFFSETS = (
    np.array([0.00270134, -0.09778530, -0.13423489], dtype=np.float64),
    np.array([0.01400000, 0.09244062, -0.13785985], dtype=np.float64),
    np.array([-0.00850000, -0.14850305, -0.04749628], dtype=np.float64),
    np.array([-0.00850000, 0.14041879, -0.06946962], dtype=np.float64),
)

PINOCCHIO_FROM_MUJOCO = np.array([6, 7, 8, 0, 1, 2, 9, 10, 11, 3, 4, 5], dtype=np.int32)
MUJOCO_FROM_PINOCCHIO = np.array([3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8], dtype=np.int32)


@dataclass
class PinocchioIK:
    max_iterations: int = 20
    eps: float = 1e-6
    damping: float = 1e-4
    foot_weight: float = 1.0
    posture_weight: float = 1e-2

    def __post_init__(self) -> None:
        self.robot = RobotWrapper.BuildFromURDF(str(URDF_PATH), [str(PINOCCHIO_MESH_ROOT)])
        self.model = self.robot.model
        self.foot_ids = self._add_foot_frames()
        self.data = self.model.createData()
        self.joint_id = {name: i for i, name in enumerate(PINOCCHIO_JOINT_ORDER)}

    def _add_foot_frames(self) -> list[int]:
        foot_ids: list[int] = []
        for calf_frame_name, foot_frame_name, offset in zip(PINOCCHIO_FOOT_FRAMES, FOOT_FRAME_NAMES, FOOT_OFFSETS):
            calf_frame_id = self.model.getFrameId(calf_frame_name)
            calf_frame = self.model.frames[calf_frame_id]
            foot_frame = pin.Frame(
                foot_frame_name,
                calf_frame.parentJoint,
                calf_frame_id,
                pin.SE3(np.eye(3), offset),
                pin.FrameType.OP_FRAME,
            )
            foot_ids.append(self.model.addFrame(foot_frame))
        return foot_ids

    def solve(
        self,
        target_feet: np.ndarray,
        seed_mujoco: np.ndarray,
        base_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        q0 = self._mujo_to_pin(seed_mujoco)
        q = q0.copy()
        target = np.asarray(target_feet, dtype=np.float64).reshape(3, 4)
        target_local = target.T
        for _ in range(self.max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = np.array([self.data.oMf[fid].translation for fid in self.foot_ids], dtype=np.float64)
            current_local = current
            error = (target_local - current_local).reshape(-1)
            if np.linalg.norm(error) < self.eps:
                break
            J = self._foot_jacobian(q)
            weighted_J = np.sqrt(self.foot_weight) * J
            weighted_error = np.sqrt(self.foot_weight) * error
            lhs = weighted_J.T @ weighted_J + self.damping * np.eye(self.model.nv)
            rhs = weighted_J.T @ weighted_error - self.posture_weight * (q - q0)
            dq = np.linalg.solve(lhs, rhs)
            dq = np.clip(dq, -0.15, 0.15)
            q = pin.integrate(self.model, q, dq)
        return self._pin_to_mujo(q)

    def _foot_jacobian(self, q: np.ndarray) -> np.ndarray:
        J = np.zeros((12, self.model.nv), dtype=np.float64)
        for i, frame_id in enumerate(self.foot_ids):
            frame_jac = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            J[i * 3 : (i + 1) * 3, :] = frame_jac[:3, :]
        return J

    def _mujo_to_pin(self, q_mujoco: np.ndarray) -> np.ndarray:
        q_mujoco = np.asarray(q_mujoco, dtype=np.float64).reshape(12)
        return q_mujoco[PINOCCHIO_FROM_MUJOCO].copy()

    def _pin_to_mujo(self, q_pin: np.ndarray) -> np.ndarray:
        q_pin = np.asarray(q_pin, dtype=np.float64).reshape(12)
        return q_pin[MUJOCO_FROM_PINOCCHIO].astype(np.float32)
