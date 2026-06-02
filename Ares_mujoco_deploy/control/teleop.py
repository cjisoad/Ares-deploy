from __future__ import annotations

from dataclasses import dataclass, field
import threading

import glfw
import numpy as np


@dataclass
class VelocityCommand:
    horizontal_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    yaw_rate: float = 0.0

    def copy(self) -> VelocityCommand:
        return VelocityCommand(self.horizontal_velocity.copy(), float(self.yaw_rate))


class VelocityTeleop:
    def __init__(
        self,
        step: float = 0.60,
        max_speed: float = 1.20,
        yaw_step: float = 0.45,
        max_yaw_rate: float = 1.20,
    ) -> None:
        self.step = step
        self.max_speed = max_speed
        self.yaw_step = yaw_step
        self.max_yaw_rate = max_yaw_rate
        self._lock = threading.Lock()
        self._command = VelocityCommand()

    def handle_key(self, key: int) -> None:
        with self._lock:
            if key == glfw.KEY_W:
                self._command.horizontal_velocity[0] = self._clip(
                    self._command.horizontal_velocity[0] + self.step
                )
            elif key == glfw.KEY_X:
                self._command.horizontal_velocity[0] = self._clip(
                    self._command.horizontal_velocity[0] - self.step
                )
            elif key == glfw.KEY_A:
                self._command.horizontal_velocity[1] = self._clip(
                    self._command.horizontal_velocity[1] + self.step
                )
            elif key == glfw.KEY_D:
                self._command.horizontal_velocity[1] = self._clip(
                    self._command.horizontal_velocity[1] - self.step
                )
            elif key == glfw.KEY_Q:
                self._command.yaw_rate = self._clip(
                    self._command.yaw_rate + self.yaw_step,
                    self.max_yaw_rate,
                )
            elif key == glfw.KEY_E:
                self._command.yaw_rate = self._clip(
                    self._command.yaw_rate - self.yaw_step,
                    self.max_yaw_rate,
                )
            elif key == glfw.KEY_SPACE:
                self._command.horizontal_velocity[:] = 0.0
                self._command.yaw_rate = 0.0

    def get_command(self) -> VelocityCommand:
        with self._lock:
            return self._command.copy()

    def set_zero(self) -> None:
        with self._lock:
            self._command.horizontal_velocity[:] = 0.0
            self._command.yaw_rate = 0.0

    def set_forward_velocity(self, velocity: float) -> None:
        with self._lock:
            self._command.horizontal_velocity[0] = self._clip(velocity)

    def set_lateral_velocity(self, velocity: float) -> None:
        with self._lock:
            self._command.horizontal_velocity[1] = self._clip(velocity)

    def set_yaw_rate(self, yaw_rate: float) -> None:
        with self._lock:
            self._command.yaw_rate = self._clip(yaw_rate, self.max_yaw_rate)

    def _clip(self, value: float, limit: float | None = None) -> float:
        bound = self.max_speed if limit is None else limit
        return float(np.clip(value, -bound, bound))
