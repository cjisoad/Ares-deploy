from __future__ import annotations

import numpy as np

from control.config import PositionControlConfig


class GaitController:
    def __init__(self, config: PositionControlConfig) -> None:
        self.config = config

    def phase_index(self, ticks: int) -> int:
        phase_time = ticks % self.config.phase_length
        phase_sum = 0
        for index, phase_ticks in enumerate(self.config.phase_ticks):
            phase_sum += int(phase_ticks)
            if phase_time < phase_sum:
                return index
        return len(self.config.phase_ticks) - 1

    def subphase_ticks(self, ticks: int) -> int:
        phase_time = ticks % self.config.phase_length
        phase_sum = 0
        for phase_ticks in self.config.phase_ticks:
            phase_sum += int(phase_ticks)
            if phase_time < phase_sum:
                return phase_time - phase_sum + int(phase_ticks)
        return 0

    def contacts(self, ticks: int) -> np.ndarray:
        return self.config.contact_phases[:, self.phase_index(ticks)]

    def swing_phase(self, ticks: int) -> float:
        return float(np.clip(self.subphase_ticks(ticks) / self.config.swing_ticks, 0.0, 1.0))
