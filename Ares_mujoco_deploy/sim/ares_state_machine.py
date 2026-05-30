from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

import numpy as np

from control.config import PositionControlCommand, PositionControlConfig
from control.position_controller import PositionController
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, CROUCH_POSE, STAND_POSE


class AresState(Enum):
    DROPPING = "dropping"
    INITIAL = "initial"
    STANDING = "standing"
    STAND = "stand"
    POSITION = "position"
    CROUCH = "crouch"


@dataclass
class AresStateMachineConfig:
    drop_duration: float = 2.0
    stand_duration: float = 3.0
    kp: float = 45.0
    kd: float = 2.0
    position: PositionControlConfig = field(default_factory=PositionControlConfig)


class AresStateMachine:
    def __init__(self, sim: AresMuJoCoSimulation, config: AresStateMachineConfig | None = None) -> None:
        self.sim = sim
        self.config = config or AresStateMachineConfig()
        self.state = AresState.DROPPING
        self.state_start_time = 0.0
        self.transition_start_pos = CROUCH_POSE.copy()
        self.commanded_pos = CROUCH_POSE.copy()
        self.position_command = PositionControlCommand.zero()
        self.position_controller = PositionController(sim.model, self.config.position)
        self._set_zero_torque(CROUCH_POSE)

    def request_stand(self) -> bool:
        if self.state not in (AresState.INITIAL, AresState.STAND, AresState.CROUCH):
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
            target, _contacts = self.position_controller.step(
                self.sim.step_count,
                self.sim.get_state()["joint_pos"],
                self.position_command,
            )
            self.sim.set_mit_command(
                kp=np.full(self.sim.dof_num, self.config.position.kp, dtype=np.float32),
                q_des=target,
                kd=np.full(self.sim.dof_num, self.config.position.kd, dtype=np.float32),
            )
            return self.sim.step()

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
        self._enter(AresState.POSITION)

    def set_position_command(self, command: PositionControlCommand) -> None:
        self.position_command = command

    def crouch(self) -> None:
        self._enter(AresState.CROUCH)

    def request_crouch(self) -> None:
        self.crouch()

    @staticmethod
    def _transition_alpha(elapsed: float, duration: float) -> float:
        if duration <= 0.0:
            return 1.0
        x = float(np.clip(elapsed / duration, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)
