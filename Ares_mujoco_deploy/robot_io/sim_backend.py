from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.ares_mujoco_simulation import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_TORQUE_LIMIT,
    MODEL_PATH,
    AresMuJoCoSimulation,
)


class MuJoCoRobotBackend:
    """Adapter that exposes AresMuJoCoSimulation through the robot backend API."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        use_viewer: bool = True,
        base_height: float = DEFAULT_BASE_HEIGHT,
        torque_limit: float = DEFAULT_TORQUE_LIMIT,
        initial_joint_pos: np.ndarray | None = None,
        verbose: bool = True,
        key_callback=None,
        show_torque_overlay: bool = False,
    ) -> None:
        self.sim = AresMuJoCoSimulation(
            model_path=model_path,
            use_viewer=use_viewer,
            base_height=base_height,
            torque_limit=torque_limit,
            initial_joint_pos=initial_joint_pos,
            verbose=verbose,
            key_callback=key_callback,
            show_torque_overlay=show_torque_overlay,
        )

    @property
    def model(self):
        return self.sim.model

    @property
    def dof_num(self) -> int:
        return self.sim.dof_num

    @property
    def step_count(self) -> int:
        return self.sim.step_count

    @property
    def timestamp(self) -> float:
        return self.sim.timestamp

    @property
    def torque_limit(self) -> float:
        return self.sim.torque_limit

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.close()

    def close(self) -> None:
        self.sim.close()

    def is_running(self) -> bool:
        return self.sim.is_running()

    def set_mit_command(
        self,
        kp: np.ndarray,
        q_des: np.ndarray,
        kd: np.ndarray,
        dq_des: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None:
        self.sim.set_mit_command(kp=kp, q_des=q_des, kd=kd, dq_des=dq_des, tau_ff=tau_ff)

    def get_state(self) -> dict[str, np.ndarray | float]:
        return self.sim.get_state()

    def step(self) -> dict[str, np.ndarray | float]:
        return self.sim.step()

    def __enter__(self) -> "MuJoCoRobotBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
