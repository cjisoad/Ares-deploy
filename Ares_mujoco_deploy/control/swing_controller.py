from __future__ import annotations

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig
from control.stance_controller import yaw_matrix


class SwingController:
    def __init__(self, config: PositionControlConfig, default_stance: np.ndarray) -> None:
        self.config = config
        self.default_stance = np.asarray(default_stance, dtype=np.float32).reshape(3, 4)

    def touchdown_location(self, leg_index: int, command: PositionControlCommand) -> np.ndarray:
        delta_xy = (
            self.config.alpha
            * self.config.stance_ticks
            * self.config.dt
            * command.horizontal_velocity
        )
        delta_p = np.array([delta_xy[0], delta_xy[1], 0.0], dtype=np.float32)
        yaw = self.config.beta * self.config.stance_ticks * self.config.dt * command.yaw_rate
        return yaw_matrix(yaw) @ self.default_stance[:, leg_index] + delta_p

    def swing_height(self, swing_phase: float) -> float:
        if swing_phase < 0.5:
            return self.config.z_clearance * swing_phase / 0.5
        return self.config.z_clearance * (1.0 - (swing_phase - 0.5) / 0.5)

    def next_foot_location(
        self,
        swing_phase: float,
        leg_index: int,
        foot_locations: np.ndarray,
        command: PositionControlCommand,
    ) -> np.ndarray:
        phase = float(np.clip(swing_phase, 0.0, 1.0))
        foot_location = foot_locations[:, leg_index]
        touchdown = self.touchdown_location(leg_index, command)
        time_left = max(self.config.dt, self.config.dt * self.config.swing_ticks * (1.0 - phase))
        velocity = (touchdown - foot_location) / time_left * np.array([1.0, 1.0, 0.0], dtype=np.float32)
        next_location = foot_location * np.array([1.0, 1.0, 0.0], dtype=np.float32) + velocity * self.config.dt
        next_location[2] = command.height + self.swing_height(phase)
        return next_location.astype(np.float32)
