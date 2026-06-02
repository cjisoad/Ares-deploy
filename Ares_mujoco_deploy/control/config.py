from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.ares_mujoco_simulation import DT


LEG_NAMES = ("lf", "rf", "lb", "rb")


@dataclass
class PositionControlCommand:
    horizontal_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    yaw_rate: float = 0.0
    height: float = 0.0

    @classmethod
    def zero(cls, height: float = 0.0) -> PositionControlCommand:
        return cls(horizontal_velocity=np.zeros(2, dtype=np.float32), height=height)


@dataclass
class PositionControlConfig:
    dt: float = DT
    overlap_time: float = 0.10
    swing_time: float = 0.20
    z_clearance: float = 0.07
    alpha: float = 3.2
    beta: float = 2.4
    z_time_constant: float = 0.08
    kp: float = 75.0
    kd: float = 3.5
    step_period: float = 0.34
    in_place_step_velocity: float = 0.20
    turn_in_place_yaw_threshold: float = 0.05
    visual_xy_gain: float = 1.8
    visual_z_gain: float = 1.8
    max_x_offset: float = 0.32
    max_y_offset: float = 0.18
    min_z_offset: float = -0.02
    max_z_offset: float = 0.16
    body_forward_bias: float = 0.05
    accumulate_foot_targets: bool = False
    yaw_feedback_gain: float = 3.0
    max_yaw_rate: float = 2.4
    enable_tilt_compensation: bool = False
    tilt_compensation_gain: float = 0.8
    max_tilt_compensation: float = 0.4

    @property
    def overlap_ticks(self) -> int:
        return max(1, int(self.overlap_time / self.dt))

    @property
    def swing_ticks(self) -> int:
        return max(1, int(self.swing_time / self.dt))

    @property
    def stance_ticks(self) -> int:
        return 2 * self.overlap_ticks + self.swing_ticks

    @property
    def phase_ticks(self) -> np.ndarray:
        return np.array(
            [self.overlap_ticks, self.swing_ticks, self.overlap_ticks, self.swing_ticks],
            dtype=np.int32,
        )

    @property
    def phase_length(self) -> int:
        return int(np.sum(self.phase_ticks))

    @property
    def contact_phases(self) -> np.ndarray:
        return np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 0],
            ],
            dtype=np.int32,
        )
