from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class MitJointCommand:
    kp: np.ndarray
    q_des: np.ndarray
    kd: np.ndarray
    dq_des: np.ndarray
    tau_ff: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        dof_num: int,
        kp: np.ndarray,
        q_des: np.ndarray,
        kd: np.ndarray,
        dq_des: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> "MitJointCommand":
        return cls(
            kp=np.asarray(kp, dtype=np.float32).reshape(dof_num),
            q_des=np.asarray(q_des, dtype=np.float32).reshape(dof_num),
            kd=np.asarray(kd, dtype=np.float32).reshape(dof_num),
            dq_des=np.zeros(dof_num, dtype=np.float32)
            if dq_des is None
            else np.asarray(dq_des, dtype=np.float32).reshape(dof_num),
            tau_ff=np.zeros(dof_num, dtype=np.float32)
            if tau_ff is None
            else np.asarray(tau_ff, dtype=np.float32).reshape(dof_num),
        )


@dataclass
class RobotState:
    time: float
    base_pos: np.ndarray
    base_rpy: np.ndarray
    base_omega: np.ndarray
    base_acc: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_tau: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "time": float(self.time),
            "base_pos": self.base_pos.astype(np.float32),
            "base_rpy": self.base_rpy.astype(np.float32),
            "base_omega": self.base_omega.astype(np.float32),
            "base_acc": self.base_acc.astype(np.float32),
            "joint_pos": self.joint_pos.astype(np.float32),
            "joint_vel": self.joint_vel.astype(np.float32),
            "joint_tau": self.joint_tau.astype(np.float32),
        }


@runtime_checkable
class RobotBackend(Protocol):
    model: Any
    dof_num: int
    step_count: int
    timestamp: float

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def is_running(self) -> bool: ...

    def set_mit_command(
        self,
        kp: np.ndarray,
        q_des: np.ndarray,
        kd: np.ndarray,
        dq_des: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None: ...

    def get_state(self) -> dict[str, np.ndarray | float]: ...

    def step(self) -> dict[str, np.ndarray | float]: ...
