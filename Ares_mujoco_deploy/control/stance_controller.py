from __future__ import annotations

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig


def yaw_matrix(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


class StanceController:
    def __init__(self, config: PositionControlConfig) -> None:
        self.config = config

    def next_foot_location(
        self,
        leg_index: int,
        foot_locations: np.ndarray,
        command: PositionControlCommand,
    ) -> np.ndarray:
        foot_location = foot_locations[:, leg_index]
        z_velocity = (command.height - foot_location[2]) / self.config.z_time_constant
        delta_p = np.array(
            [
                -command.horizontal_velocity[0],
                -command.horizontal_velocity[1],
                z_velocity,
            ],
            dtype=np.float32,
        ) * self.config.dt
        delta_r = yaw_matrix(-command.yaw_rate * self.config.dt)
        return delta_r @ foot_location + delta_p
