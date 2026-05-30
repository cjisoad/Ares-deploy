from __future__ import annotations

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig
from control.gait import GaitController
from control.kinematics import AresKinematics, default_foot_locations
from control.stance_controller import StanceController
from control.swing_controller import SwingController
from sim.ares_mujoco_simulation import STAND_POSE


class PositionController:
    def __init__(self, model, config: PositionControlConfig | None = None) -> None:
        self.config = config or PositionControlConfig()
        self.kinematics = AresKinematics(model)
        self.gait = GaitController(self.config)
        self.default_stance = default_foot_locations(model)
        self.stance_controller = StanceController(self.config)
        self.swing_controller = SwingController(self.config, self.default_stance)
        self.foot_locations = self.default_stance.copy()
        self.joint_targets = STAND_POSE.copy()

    def reset(self, joint_pos: np.ndarray | None = None) -> None:
        seed = STAND_POSE if joint_pos is None else joint_pos
        self.foot_locations = self.kinematics.foot_positions(seed)
        self.joint_targets = np.asarray(seed, dtype=np.float32).reshape(12).copy()
        self.ticks = 0

    def set_foot_pattern(self, foot_locations: np.ndarray, joint_pos: np.ndarray | None = None) -> np.ndarray:
        seed = self.joint_targets if joint_pos is None else joint_pos
        self.foot_locations = np.asarray(foot_locations, dtype=np.float32).reshape(3, 4).copy()
        self.joint_targets = self.kinematics.inverse_kinematics(self.foot_locations, seed)
        return self.joint_targets.copy()

    def step(
        self,
        ticks: int,
        current_joint_pos: np.ndarray,
        command: PositionControlCommand,
    ) -> tuple[np.ndarray, np.ndarray]:
        contacts = self.gait.contacts(ticks)
        swing_phase = self.gait.swing_phase(ticks)
        next_locations = self.foot_locations.copy()
        self.ticks = ticks
        for leg_index in range(4):
            if contacts[leg_index]:
                next_locations[:, leg_index] = self.stance_controller.next_foot_location(
                    leg_index,
                    self.foot_locations,
                    command,
                )
            else:
                next_locations[:, leg_index] = self.swing_controller.next_foot_location(
                    swing_phase,
                    leg_index,
                    self.foot_locations,
                    command,
                )
        self.foot_locations = next_locations
        self.joint_targets = self.kinematics.inverse_kinematics(next_locations, current_joint_pos)
        return self.joint_targets.copy(), contacts.copy()
