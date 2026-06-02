from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig
from control.position_controller import PositionController
from robot_io.types import RobotBackend
from sim.ares_mujoco_simulation import CROUCH_POSE, STAND_POSE


class AresState(Enum):
    DROPPING = "dropping"
    INITIAL = "initial"
    STANDING = "standing"
    STAND = "stand"
    POSITION = "position"
    CROUCHING = "crouching"
    CROUCH = "crouch"


@dataclass
class AresStateMachineConfig:
    drop_duration: float = 2.0
    stand_duration: float = 3.0
    crouch_duration: float = 3.0
    kp: float = 45.0
    kd: float = 2.0
    position: PositionControlConfig = field(default_factory=PositionControlConfig)


class AresStateMachine:
    def __init__(self, sim: RobotBackend, config: AresStateMachineConfig | None = None) -> None:
        self.sim = sim
        self.config = config or AresStateMachineConfig()
        self.state = AresState.DROPPING
        self.state_start_time = 0.0
        self.transition_start_pos = CROUCH_POSE.copy()
        self.commanded_pos = CROUCH_POSE.copy()
        self.position_command = PositionControlCommand.zero()
        self.position_controller = PositionController(sim.model, self.config.position)
        self.last_position_target = STAND_POSE.copy()
        self.last_position_contacts = np.ones(4, dtype=np.int32)
        self.position_heading_target = 0.0
        self.position_unwrapped_yaw = 0.0
        self.position_last_wrapped_yaw = 0.0
        self.position_yaw_command_active = False
        self.last_effective_yaw_rate = 0.0
        self._set_zero_torque(CROUCH_POSE)

    def request_stand(self) -> bool:
        if self.state not in (AresState.INITIAL, AresState.STAND, AresState.CROUCH, AresState.POSITION, AresState.CROUCHING):
            return False
        self.transition_start_pos = self.sim.get_state()["joint_pos"].copy()
        self.commanded_pos = self.transition_start_pos.copy()
        self._enter(AresState.STANDING)
        return True

    def request_position(self, command: PositionControlCommand | None = None) -> bool:
        if self.state != AresState.STAND:
            return False
        self.enter_position_mode(command)
        return True

    def step(self) -> dict[str, np.ndarray | float]:
        if self.state == AresState.DROPPING:
            self._set_zero_torque(CROUCH_POSE)
            state = self.sim.step()
            if self._state_elapsed() >= self.config.drop_duration:
                self._enter(AresState.INITIAL)
            return state

        if self.state == AresState.INITIAL:
            self._set_hold_command(CROUCH_POSE)
            return self.sim.step()

        if self.state == AresState.STANDING:
            alpha = self._transition_alpha(self._state_elapsed(), self.config.stand_duration)
            self.commanded_pos = (1.0 - alpha) * self.transition_start_pos + alpha * STAND_POSE
            self._set_hold_command(self.commanded_pos)
            state = self.sim.step()
            if alpha >= 1.0:
                self.commanded_pos = STAND_POSE.copy()
                self._enter(AresState.STAND)
            return state

        if self.state == AresState.POSITION:
            sim_state = self.sim.get_state()
            current_joint_pos = sim_state["joint_pos"]
            position_command = self._stabilized_position_command(sim_state)
            if np.allclose(position_command.horizontal_velocity, 0.0):
                target, _contacts = self.position_controller.in_place_step(
                    self.sim.step_count,
                    current_joint_pos,
                    position_command,
                    sim_state["base_rpy"],
                )
            else:
                target, _contacts = self.position_controller.step(
                    self.sim.step_count,
                    current_joint_pos,
                    position_command,
                    sim_state["base_rpy"],
                )
            self.last_position_target = target.copy()
            self.last_position_contacts = _contacts.copy()
            self.sim.set_mit_command(
                kp=np.full(self.sim.dof_num, self.config.position.kp, dtype=np.float32),
                q_des=target,
                kd=np.full(self.sim.dof_num, self.config.position.kd, dtype=np.float32),
            )
            return self.sim.step()

        if self.state == AresState.CROUCHING:
            alpha = self._transition_alpha(self._state_elapsed(), self.config.crouch_duration)
            self.commanded_pos = (1.0 - alpha) * self.transition_start_pos + alpha * CROUCH_POSE
            self._set_hold_command(self.commanded_pos)
            state = self.sim.step()
            if alpha >= 1.0:
                self.commanded_pos = CROUCH_POSE.copy()
                self._enter(AresState.CROUCH)
            return state

        if self.state == AresState.CROUCH:
            self._set_hold_command(CROUCH_POSE)
            return self.sim.step()

        self._set_hold_command(STAND_POSE)
        return self.sim.step()

    def _enter(self, state: AresState) -> None:
        self.state = state
        self.state_start_time = self.sim.timestamp

    def _state_elapsed(self) -> float:
        return self.sim.timestamp - self.state_start_time

    def _set_zero_torque(self, q_des: np.ndarray) -> None:
        zero = np.zeros(self.sim.dof_num, dtype=np.float32)
        self.sim.set_mit_command(kp=zero, q_des=q_des, kd=zero)

    def _set_hold_command(self, q_des: np.ndarray) -> None:
        kp = np.full(self.sim.dof_num, self.config.kp, dtype=np.float32)
        kd = np.full(self.sim.dof_num, self.config.kd, dtype=np.float32)
        self.sim.set_mit_command(kp=kp, q_des=q_des, kd=kd)

    def enter_position_mode(self, command: PositionControlCommand | None = None) -> None:
        default_height = float(self.position_controller.default_stance[2].mean())
        self.position_command = command or PositionControlCommand.zero(height=default_height)
        if self.position_command.height == 0.0:
            self.position_command.height = default_height
        self.position_controller.reset(self.sim.get_state()["joint_pos"])
        current_yaw = float(self.sim.get_state()["base_rpy"][2])
        self.position_heading_target = current_yaw
        self.position_unwrapped_yaw = current_yaw
        self.position_last_wrapped_yaw = current_yaw
        self.position_yaw_command_active = abs(float(self.position_command.yaw_rate)) > self.config.position.turn_in_place_yaw_threshold
        self.last_effective_yaw_rate = float(self.position_command.yaw_rate)
        self._enter(AresState.POSITION)

    def set_position_command(self, command: PositionControlCommand) -> None:
        self.position_command = command

    def crouch(self) -> None:
        self.transition_start_pos = self.sim.get_state()["joint_pos"].copy()
        self.commanded_pos = self.transition_start_pos.copy()
        self._enter(AresState.CROUCHING)

    def request_crouch(self) -> None:
        self.crouch()

    @staticmethod
    def _transition_alpha(elapsed: float, duration: float) -> float:
        if duration <= 0.0:
            return 1.0
        x = float(np.clip(elapsed / duration, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _stabilized_position_command(self, sim_state: dict[str, np.ndarray | float]) -> PositionControlCommand:
        raw = self.position_command
        current_yaw = float(np.asarray(sim_state["base_rpy"], dtype=np.float32)[2])
        yaw_delta = self._wrap_angle(current_yaw - self.position_last_wrapped_yaw)
        self.position_unwrapped_yaw += yaw_delta
        self.position_last_wrapped_yaw = current_yaw

        yaw_command_active = abs(float(raw.yaw_rate)) > self.config.position.turn_in_place_yaw_threshold
        if yaw_command_active:
            self.position_heading_target += raw.yaw_rate * self.config.position.dt
        elif self.position_yaw_command_active:
            self.position_heading_target = self.position_unwrapped_yaw
        self.position_yaw_command_active = yaw_command_active

        yaw_error = self.position_heading_target - self.position_unwrapped_yaw
        effective_yaw_rate = raw.yaw_rate + self.config.position.yaw_feedback_gain * yaw_error
        effective_yaw_rate = float(
            np.clip(
                effective_yaw_rate,
                -self.config.position.max_yaw_rate,
                self.config.position.max_yaw_rate,
            )
        )
        self.last_effective_yaw_rate = effective_yaw_rate
        return PositionControlCommand(
            horizontal_velocity=raw.horizontal_velocity.copy(),
            yaw_rate=effective_yaw_rate,
            height=raw.height,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
