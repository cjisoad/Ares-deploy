from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robot_io.rs02.config import require_section
from robot_io.rs02.controller import MotorConfig, RS02MitController
from robot_io.rs02.protocol import (
    KD_MAX,
    KP_MAX,
    T_MAX,
    TimedMotorFeedback,
    MotorFeedback,
    RS02MitCommand,
    clamp,
)
from robot_io.types import MitJointCommand, RobotState
from sim.ares_mujoco_simulation import DEFAULT_BASE_QUAT, JOINT_ORDER, MODEL_PATH


@dataclass(frozen=True)
class JointMotorMapping:
    joint_name: str
    motor_name: str
    motor_id: int
    enabled: bool
    sign: float
    zero_offset: float
    joint_min: float | None
    joint_max: float | None
    motor_min: float | None
    motor_max: float | None

    def joint_to_motor_angle(self, angle: float) -> float:
        return self.sign * angle + self.zero_offset

    def motor_to_joint_angle(self, angle: float) -> float:
        return self.sign * (angle - self.zero_offset)

    def joint_to_motor_velocity(self, velocity: float) -> float:
        return self.sign * velocity

    def motor_to_joint_velocity(self, velocity: float) -> float:
        return self.sign * velocity

    def joint_to_motor_torque(self, torque: float) -> float:
        return self.sign * torque

    def motor_to_joint_torque(self, torque: float) -> float:
        return self.sign * torque

    def clamp_joint_angle(self, angle: float) -> float:
        if self.joint_min is not None:
            angle = max(self.joint_min, angle)
        if self.joint_max is not None:
            angle = min(self.joint_max, angle)
        return angle


def build_mappings(config: dict[str, Any]) -> list[JointMotorMapping]:
    joints = require_section(config, "joints")
    mappings: list[JointMotorMapping] = []
    seen_motor_ids: set[int] = set()
    for joint_name in JOINT_ORDER:
        cfg = joints.get(joint_name)
        if not isinstance(cfg, dict):
            raise ValueError(f"Missing RS02 joint mapping for {joint_name}")
        motor_id = int(cfg["motor_id"]) & 0xFF
        if motor_id == 0:
            raise ValueError(f"Invalid motor ID 0 for {joint_name}")
        if motor_id in seen_motor_ids:
            raise ValueError(f"Duplicate motor ID 0x{motor_id:02X}")
        seen_motor_ids.add(motor_id)
        mappings.append(
            JointMotorMapping(
                joint_name=joint_name,
                motor_name=str(cfg.get("motor_name", joint_name)),
                motor_id=motor_id,
                enabled=bool(cfg.get("enabled", False)),
                sign=float(cfg.get("sign", 1.0)),
                zero_offset=float(cfg.get("zero_offset", 0.0)),
                joint_min=None if cfg.get("joint_min") is None else float(cfg["joint_min"]),
                joint_max=None if cfg.get("joint_max") is None else float(cfg["joint_max"]),
                motor_min=None if cfg.get("motor_min") is None else float(cfg["motor_min"]),
                motor_max=None if cfg.get("motor_max") is None else float(cfg["motor_max"]),
            )
        )
    return mappings


def build_motor_configs(mappings: list[JointMotorMapping], default_kd: float) -> dict[int, MotorConfig]:
    result: dict[int, MotorConfig] = {}
    for mapping in mappings:
        result[mapping.motor_id] = MotorConfig(
            name=mapping.motor_name,
            motor_id=mapping.motor_id,
            enabled=mapping.enabled,
            default_command=RS02MitCommand(kd=default_kd),
            angle_min=mapping.motor_min,
            angle_max=mapping.motor_max,
        )
    return result


class RS02HardwareBackend:
    """12-joint Ares backend for RS02 MIT motors.

    Dry-run is the default. Live mode is only used when explicitly requested by
    the caller.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        live: bool = False,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.config = config
        self.live = live
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.dof_num = 12
        self.step_count = 0
        self.timestamp = 0.0
        self.mappings = build_mappings(config)
        self.mapping_by_motor_id = {mapping.motor_id: mapping for mapping in self.mappings}
        self.mapping_by_joint_name = {mapping.joint_name: mapping for mapping in self.mappings}
        self.command = MitJointCommand.from_arrays(
            self.dof_num,
            kp=np.zeros(self.dof_num, dtype=np.float32),
            q_des=np.zeros(self.dof_num, dtype=np.float32),
            kd=np.zeros(self.dof_num, dtype=np.float32),
        )
        self.joint_pos = np.zeros(self.dof_num, dtype=np.float32)
        self.joint_vel = np.zeros(self.dof_num, dtype=np.float32)
        self.joint_tau = np.zeros(self.dof_num, dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_pos[2] = float(require_section(config, "state").get("base_height", 0.0))
        self.base_rpy = np.zeros(3, dtype=np.float32)
        self.base_omega = np.zeros(3, dtype=np.float32)
        self.base_acc = np.zeros(3, dtype=np.float32)
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._thread_error: BaseException | None = None

        interface = require_section(config, "interface")
        safety = require_section(config, "safety")
        self.rate_hz = float(interface.get("rate_hz", 100.0))
        self.dt = 1.0 / max(1.0, self.rate_hz)
        self.feedback_timeout = float(safety.get("feedback_timeout", 0.25))
        self.command_timeout = float(safety.get("command_timeout", 0.25))
        self.default_kd = float(safety.get("default_damping_kd", 0.2))
        self.last_command_time = time.monotonic()
        self.controller: RS02MitController | None = None
        if self.live:
            self.controller = RS02MitController(
                config=interface,
                motor_configs=build_motor_configs(self.mappings, self.default_kd),
            )

    def start(self) -> None:
        self._running = True
        self.last_command_time = time.monotonic()
        if not self.live:
            return
        assert self.controller is not None
        self._thread_error = None
        self._thread = threading.Thread(target=self._run_controller, daemon=True)
        self._thread.start()
        self._wait_until_live_started(float(require_section(self.config, "safety").get("startup_timeout", 3.0)))

    def stop(self) -> None:
        self._running = False
        if self.controller is not None:
            self.controller.running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.controller is not None:
            self.controller.stop()
        self._raise_thread_error()

    def close(self) -> None:
        self.stop()

    def is_running(self) -> bool:
        return self._running

    def set_mit_command(
        self,
        kp: np.ndarray,
        q_des: np.ndarray,
        kd: np.ndarray,
        dq_des: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None:
        command = MitJointCommand.from_arrays(self.dof_num, kp, q_des, kd, dq_des, tau_ff)
        with self._lock:
            self.command = command
            self.last_command_time = time.monotonic()
        if self.live and self.controller is not None:
            for joint_index, mapping in enumerate(self.mappings):
                if not mapping.enabled:
                    continue
                self.controller.set_motor(mapping.motor_id, self._joint_command_to_motor(mapping, joint_index, command))

    def get_state(self) -> dict[str, np.ndarray | float]:
        self._raise_thread_error()
        if self.live and self.controller is not None:
            self._update_live_feedback()
        state = RobotState(
            time=float(self.timestamp),
            base_pos=self.base_pos.copy(),
            base_rpy=self.base_rpy.copy(),
            base_omega=self.base_omega.copy(),
            base_acc=self.base_acc.copy(),
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
            joint_tau=self.joint_tau.copy(),
        )
        return state.as_dict()

    def step(self) -> dict[str, np.ndarray | float]:
        self._raise_thread_error()
        if self.live and time.monotonic() - self.last_command_time > self.command_timeout:
            self._send_damping_command()
        if not self.live:
            self._step_dry_run()
        else:
            self._update_live_feedback()
        self.step_count += 1
        self.timestamp = self.step_count * self.dt
        return self.get_state()

    def _joint_command_to_motor(
        self,
        mapping: JointMotorMapping,
        joint_index: int,
        command: MitJointCommand,
    ) -> RS02MitCommand:
        q_des = mapping.clamp_joint_angle(float(command.q_des[joint_index]))
        return RS02MitCommand(
            torque=clamp(mapping.joint_to_motor_torque(float(command.tau_ff[joint_index])), -T_MAX, T_MAX),
            angle=mapping.joint_to_motor_angle(q_des),
            speed=mapping.joint_to_motor_velocity(float(command.dq_des[joint_index])),
            kp=clamp(float(command.kp[joint_index]), 0.0, KP_MAX),
            kd=clamp(float(command.kd[joint_index]), 0.0, KD_MAX),
        )

    def _step_dry_run(self) -> None:
        with self._lock:
            command = self.command
        alpha = float(require_section(self.config, "dry_run").get("tracking_alpha", 1.0))
        next_pos = self.joint_pos + alpha * (command.q_des - self.joint_pos)
        self.joint_vel = ((next_pos - self.joint_pos) / max(self.dt, 1.0e-6)).astype(np.float32)
        self.joint_pos = next_pos.astype(np.float32)
        self.joint_tau = (
            command.kp * (command.q_des - self.joint_pos)
            + command.kd * (command.dq_des - self.joint_vel)
            + command.tau_ff
        ).astype(np.float32)

    def _update_live_feedback(self) -> None:
        assert self.controller is not None
        now = time.monotonic()
        for index, mapping in enumerate(self.mappings):
            timed = self.controller.get_timed_feedback(mapping.motor_id)
            if timed is None:
                continue
            if now - timed.updated_at > self.feedback_timeout:
                continue
            self._apply_feedback(index, mapping, timed)

    def _apply_feedback(self, index: int, mapping: JointMotorMapping, timed: TimedMotorFeedback) -> None:
        value = timed.value
        self.joint_pos[index] = mapping.motor_to_joint_angle(value.angle)
        self.joint_vel[index] = mapping.motor_to_joint_velocity(value.speed)
        self.joint_tau[index] = mapping.motor_to_joint_torque(value.torque)

    def _send_damping_command(self) -> None:
        if self.controller is None:
            return
        for mapping in self.mappings:
            if not mapping.enabled:
                continue
            self.controller.set_motor(mapping.motor_id, RS02MitCommand(kd=self.default_kd))

    def _run_controller(self) -> None:
        assert self.controller is not None
        try:
            self.controller.start()
        except BaseException as exc:
            self._thread_error = exc
            self._running = False
            self.controller.running.clear()

    def _wait_until_live_started(self, timeout: float) -> None:
        assert self.controller is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_thread_error()
            if self.controller.bus is not None and self.controller.running.is_set():
                return
            time.sleep(0.01)
        raise TimeoutError("RS02 controller did not start before timeout")

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError("RS02 hardware backend thread failed") from self._thread_error

    def __enter__(self) -> "RS02HardwareBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
