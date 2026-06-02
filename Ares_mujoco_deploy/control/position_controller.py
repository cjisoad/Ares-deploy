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
        self.last_contacts = np.ones(4, dtype=np.int32)
        self.last_foot_targets = self.default_stance.copy()

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
        base_rpy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ticks = self._periodic_ticks(ticks)
        contacts = self.gait.contacts(ticks)
        swing_phase = self.gait.swing_phase(ticks)
        reference_stance = self._reference_stance()
        current_locations = self._current_foot_reference(reference_stance)
        next_locations = current_locations.copy()
        in_place_velocity = 0.0 if abs(command.yaw_rate) > self.config.turn_in_place_yaw_threshold else self.config.in_place_step_velocity
        in_place_command = PositionControlCommand(
            horizontal_velocity=np.array([in_place_velocity, 0.0], dtype=np.float32),
            yaw_rate=command.yaw_rate,
            height=command.height,
        )
        for leg_index in range(4):
            if contacts[leg_index]:
                next_locations[:, leg_index] = self.stance_controller.next_foot_location(
                    leg_index,
                    current_locations,
                    in_place_command,
                )
            else:
                next_locations[:, leg_index] = self.swing_controller.next_foot_location(
                    swing_phase,
                    leg_index,
                    current_locations,
                    in_place_command,
                    reference_stance,
                )
            next_locations[:, leg_index] = self._scaled_target(leg_index, next_locations[:, leg_index])
        next_locations = self._tilt_compensated_targets(next_locations, base_rpy)
        self.foot_locations = next_locations
        self.last_contacts = contacts.copy()
        self.last_foot_targets = next_locations.copy()
        self.joint_targets = self.pin_ik.solve(next_locations, current_joint_pos)
        return self.joint_targets.copy(), contacts.copy()

    def step(
        self,
        ticks: int,
        current_joint_pos: np.ndarray,
        command: PositionControlCommand,
        base_rpy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ticks = self._periodic_ticks(ticks)
        contacts = self.gait.contacts(ticks)
        swing_phase = self.gait.swing_phase(ticks)
        reference_stance = self._reference_stance()
        current_locations = self._current_foot_reference(reference_stance)
        next_locations = current_locations.copy()
        self.ticks = ticks
        for leg_index in range(4):
            if contacts[leg_index]:
                next_locations[:, leg_index] = self.stance_controller.next_foot_location(
                    leg_index,
                    current_locations,
                    command,
                )
            else:
                next_locations[:, leg_index] = self.swing_controller.next_foot_location(
                    swing_phase,
                    leg_index,
                    current_locations,
                    command,
                    reference_stance,
                )
            next_locations[:, leg_index] = self._scaled_target(leg_index, next_locations[:, leg_index])
        next_locations = self._tilt_compensated_targets(next_locations, base_rpy)
        self.foot_locations = next_locations
        self.last_contacts = contacts.copy()
        self.last_foot_targets = next_locations.copy()
        self.joint_targets = self.pin_ik.solve(next_locations, current_joint_pos)
        return self.joint_targets.copy(), contacts.copy()

    def _periodic_ticks(self, ticks: int) -> int:
        if self.config.step_period <= 0.0:
            return ticks
        period_ticks = max(1, int(self.config.step_period / self.config.dt))
        phase = (ticks % period_ticks) / period_ticks
        return int(phase * self.gait.config.phase_length)

    def _reference_stance(self) -> np.ndarray:
        reference = self.default_stance.copy()
        reference[0, :] -= self.config.body_forward_bias
        return reference

    def _current_foot_reference(self, reference_stance: np.ndarray) -> np.ndarray:
        if self.config.accumulate_foot_targets:
            return self.foot_locations.copy()
        return reference_stance.copy()

    def _tilt_compensated_targets(self, foot_targets: np.ndarray, base_rpy: np.ndarray | None) -> np.ndarray:
        if not self.config.enable_tilt_compensation or base_rpy is None:
            return foot_targets
        roll, pitch = np.asarray(base_rpy, dtype=np.float32).reshape(3)[:2]
        roll_comp = self.config.tilt_compensation_gain * np.clip(
            roll,
            -self.config.max_tilt_compensation,
            self.config.max_tilt_compensation,
        )
        pitch_comp = self.config.tilt_compensation_gain * np.clip(
            pitch,
            -self.config.max_tilt_compensation,
            self.config.max_tilt_compensation,
        )
        return (self._roll_pitch_matrix(roll_comp, pitch_comp).T @ foot_targets).astype(np.float32)

    @staticmethod
    def _roll_pitch_matrix(roll: float, pitch: float) -> np.ndarray:
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float32,
        )
        return ry @ rx

    def _scaled_target(self, leg_index: int, target: np.ndarray) -> np.ndarray:
        base = self._reference_stance()[:, leg_index]
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
