from __future__ import annotations

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig
from control.gait import GaitController
from control.kinematics import AresKinematics, default_foot_locations
from control.pinocchio_ik import PinocchioIK
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
        self.pin_ik = PinocchioIK()
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

    def in_place_step(
        self,
        ticks: int,
        current_joint_pos: np.ndarray,
        command: PositionControlCommand,
    ) -> tuple[np.ndarray, np.ndarray]:
        ticks = self._periodic_ticks(ticks)
        contacts = self.gait.contacts(ticks)
        swing_phase = self.gait.swing_phase(ticks)
        next_locations = self.default_stance.copy()
        in_place_command = PositionControlCommand(
            horizontal_velocity=np.array([self.config.in_place_step_velocity, 0.0], dtype=np.float32),
            yaw_rate=0.0,
            height=command.height,
        )
        for leg_index in range(4):
            if contacts[leg_index]:
                next_locations[:, leg_index] = self.stance_controller.next_foot_location(
                    leg_index,
                    self.default_stance,
                    in_place_command,
                )
            else:
                next_locations[:, leg_index] = self.swing_controller.next_foot_location(
                    swing_phase,
                    leg_index,
                    self.default_stance,
                    in_place_command,
                )
            next_locations[:, leg_index] = self._scaled_target(leg_index, next_locations[:, leg_index])
        self.foot_locations = next_locations
        self.joint_targets = self.pin_ik.solve(next_locations, current_joint_pos)
        return self.joint_targets.copy(), contacts.copy()

    def step(
        self,
        ticks: int,
        current_joint_pos: np.ndarray,
        command: PositionControlCommand,
    ) -> tuple[np.ndarray, np.ndarray]:
        ticks = self._periodic_ticks(ticks)
        contacts = self.gait.contacts(ticks)
        swing_phase = self.gait.swing_phase(ticks)
        next_locations = self.default_stance.copy()
        self.ticks = ticks
        for leg_index in range(4):
            if contacts[leg_index]:
                next_locations[:, leg_index] = self.stance_controller.next_foot_location(
                    leg_index,
                    self.default_stance,
                    command,
                )
            else:
                next_locations[:, leg_index] = self.swing_controller.next_foot_location(
                    swing_phase,
                    leg_index,
                    self.default_stance,
                    command,
                )
            next_locations[:, leg_index] = self._scaled_target(leg_index, next_locations[:, leg_index])
        self.foot_locations = next_locations
        self.joint_targets = self.pin_ik.solve(next_locations, current_joint_pos)
        return self.joint_targets.copy(), contacts.copy()

    def _periodic_ticks(self, ticks: int) -> int:
        if self.config.step_period <= 0.0:
            return ticks
        period_ticks = max(1, int(self.config.step_period / self.config.dt))
        phase = (ticks % period_ticks) / period_ticks
        return int(phase * self.gait.config.phase_length)

    def _scaled_target(self, leg_index: int, target: np.ndarray) -> np.ndarray:
        base = self.default_stance[:, leg_index]
        scaled = np.array(
            [
                np.clip(
                    base[0] + self.config.visual_xy_gain * (target[0] - base[0]),
                    base[0] - self.config.max_x_offset,
                    base[0] + self.config.max_x_offset,
                ),
                np.clip(
                    base[1] + self.config.visual_xy_gain * (target[1] - base[1]),
                    base[1] - self.config.max_y_offset,
                    base[1] + self.config.max_y_offset,
                ),
                np.clip(
                    base[2] + self.config.visual_z_gain * (target[2] - base[2]),
                    base[2] + self.config.min_z_offset,
                    base[2] + self.config.max_z_offset,
                ),
            ],
            dtype=np.float32,
        )
        return scaled
