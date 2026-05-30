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
        step: float = 0.05,
        max_speed: float = 0.30,
        max_yaw_rate: float = 1.0,
    ) -> None:
        self.step = step
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self._lock = threading.Lock()
        self._command = VelocityCommand()

    def handle_key(self, key: int) -> None:
        with self._lock:
            if key == glfw.KEY_W:
                self._command.horizontal_velocity[0] = self._clip(
                    self._command.horizontal_velocity[0] + self.step
                )
            elif key == glfw.KEY_S:
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

    def _clip(self, value: float, limit: float | None = None) -> float:
        bound = self.max_speed if limit is None else limit
        return float(np.clip(value, -bound, bound))
