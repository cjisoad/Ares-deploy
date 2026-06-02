from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from robot_io.rs02.protocol import (
    COMM_MOTION_CONTROL,
    COMM_MOTOR_ENABLE,
    COMM_MOTOR_STOP,
    COMM_SET_SINGLE_PARAMETER,
    MIT_RUN_MODE,
    RUN_MODE,
    T_MIN,
    T_MAX,
    float_to_uint,
    parse_feedback,
    RS02MitCommand,
    TimedMotorFeedback,
    MotorFeedback,
)
from robot_io.rs02.serial_bus import RS02AtSerialBus, can


@dataclass(frozen=True)
class MotorConfig:
    name: str
    motor_id: int
    enabled: bool
    default_command: RS02MitCommand
    angle_min: float | None = None
    angle_max: float | None = None

    def validate_command(self, command: RS02MitCommand) -> RS02MitCommand | None:
        command = command.clamped()
        if self.angle_min is not None and command.angle < self.angle_min:
            return None
        if self.angle_max is not None and command.angle > self.angle_max:
            return None
        return command


class RS02Protocol:
    def __init__(self, bus: RS02AtSerialBus, motor_id: int, master_id: int):
        self.bus = bus
        self.motor_id = motor_id & 0xFF
        self.master_id = master_id & 0xFF
        self._send_lock = threading.Lock()

    def _addressed_id(self, comm_type: int) -> int:
        return (comm_type << 24) | (self.master_id << 8) | self.motor_id

    def send_frame(self, arbitration_id: int, data: bytes | bytearray | list[int]) -> None:
        if can is None:
            raise RuntimeError("python-can is not installed")
        payload = bytes(data)
        if len(payload) != 8:
            raise ValueError("RS02 frames use exactly 8 data bytes")
        msg = can.Message(
            arbitration_id=arbitration_id & 0x1FFFFFFF,
            data=payload,
            is_extended_id=True,
            is_remote_frame=False,
        )
        with self._send_lock:
            self.bus.send(msg, timeout=0.2)

    def disable(self, clear_error: bool = False) -> None:
        self.send_frame(
            self._addressed_id(COMM_MOTOR_STOP),
            bytes([1 if clear_error else 0, 0, 0, 0, 0, 0, 0, 0]),
        )

    def enable(self) -> None:
        self.send_frame(self._addressed_id(COMM_MOTOR_ENABLE), bytes(8))

    def set_run_mode(self, mode: int) -> None:
        data = bytearray(8)
        data[0] = RUN_MODE & 0xFF
        data[1] = (RUN_MODE >> 8) & 0xFF
        data[4] = mode & 0xFF
        self.send_frame(self._addressed_id(COMM_SET_SINGLE_PARAMETER), data)

    def motion_control(self, command: RS02MitCommand) -> None:
        command = command.clamped()
        torque_raw = float_to_uint(command.torque, T_MIN, T_MAX)
        arbitration_id = (COMM_MOTION_CONTROL << 24) | (torque_raw << 8) | self.motor_id
        data = bytearray(8)
        values = (
            float_to_uint(command.angle, -12.57, 12.57),
            float_to_uint(command.speed, -44.0, 44.0),
            float_to_uint(command.kp, 0.0, 500.0),
            float_to_uint(command.kd, 0.0, 5.0),
        )
        for offset, raw in zip((0, 2, 4, 6), values):
            data[offset] = (raw >> 8) & 0xFF
            data[offset + 1] = raw & 0xFF
        self.send_frame(arbitration_id, data)


class RS02MitController:
    def __init__(self, config: dict, motor_configs: dict[int, MotorConfig]):
        self.port = str(config["port"])
        self.serial_baud = int(config["serial_baud"])
        self.master_id = int(config["master_id"]) & 0xFF
        self.rate_hz = max(1.0, min(300.0, float(config["rate_hz"])))
        self.motor_configs = motor_configs
        self.motor_ids = [motor_id for motor_id, cfg in motor_configs.items() if cfg.enabled]
        if not self.motor_ids:
            raise ValueError("At least one RS02 motor must be enabled for live output")

        self.bus: RS02AtSerialBus | None = None
        self.protocols: dict[int, RS02Protocol] = {}
        self.commands: dict[int, RS02MitCommand] = {}
        for motor_id in self.motor_ids:
            command = self.motor_configs[motor_id].validate_command(self.motor_configs[motor_id].default_command)
            if command is None:
                raise ValueError(f"Default command violates motor limit: 0x{motor_id:02X}")
            self.commands[motor_id] = command
        self.feedbacks: dict[int, TimedMotorFeedback] = {}
        self.running = threading.Event()
        self.rx_thread: threading.Thread | None = None
        self._feedback_seq = 0
        self._command_lock = threading.Lock()
        self._feedback_lock = threading.Lock()

    def connect(self) -> None:
        self.bus = RS02AtSerialBus(self.port, self.serial_baud)
        self.protocols = {
            motor_id: RS02Protocol(self.bus, motor_id, self.master_id)
            for motor_id in self.motor_configs
        }

    def start(self) -> None:
        if self.bus is None:
            self.connect()
        self.running.set()
        self.rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.rx_thread.start()
        self._enter_mit_mode()
        self._control_loop()

    def stop(self) -> None:
        self.running.clear()
        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=0.3)
        self.rx_thread = None
        for motor_id in self.motor_ids:
            protocol = self.protocols.get(motor_id)
            if protocol is None:
                continue
            try:
                protocol.motion_control(RS02MitCommand())
                time.sleep(0.005)
                protocol.disable(False)
                time.sleep(0.005)
            except Exception as exc:
                print(f"RS02 stop warning motor=0x{motor_id:02X}: {exc}")
        if self.bus is not None:
            try:
                self.bus.shutdown()
            finally:
                self.bus = None

    def set_motor(self, motor_id: int, command: RS02MitCommand) -> None:
        motor_id = int(motor_id) & 0xFF
        if motor_id not in self.motor_configs:
            raise ValueError(f"Unknown RS02 motor ID: 0x{motor_id:02X}")
        if not self.motor_configs[motor_id].enabled:
            return
        validated = self.motor_configs[motor_id].validate_command(command)
        if validated is None:
            return
        with self._command_lock:
            self.commands[motor_id] = validated

    def get_feedback(self, motor_id: int) -> MotorFeedback | None:
        timed = self.get_timed_feedback(motor_id)
        return None if timed is None else timed.value

    def get_timed_feedback(self, motor_id: int) -> TimedMotorFeedback | None:
        with self._feedback_lock:
            return self.feedbacks.get(int(motor_id) & 0xFF)

    def _enter_mit_mode(self) -> None:
        for motor_id, protocol in self.protocols.items():
            protocol.disable(False)
            time.sleep(0.02)
            if not self.motor_configs[motor_id].enabled:
                continue
            protocol.set_run_mode(MIT_RUN_MODE)
            time.sleep(0.02)
            protocol.enable()
            time.sleep(0.02)

    def _control_loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_time = time.monotonic()
        try:
            while self.running.is_set():
                with self._command_lock:
                    commands = dict(self.commands)
                for motor_id, command in commands.items():
                    self.protocols[motor_id].motion_control(command)
                    time.sleep(0.002)
                next_time += period
                sleep_time = next_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.monotonic()
        finally:
            self.running.clear()

    def _recv_loop(self) -> None:
        assert self.bus is not None
        expected_ids = set(self.motor_configs)
        while self.running.is_set():
            try:
                msg = self.bus.recv(timeout=0.05)
            except Exception as exc:
                print(f"RS02 RX warning: {exc}")
                time.sleep(0.1)
                continue
            if msg is None:
                continue
            feedback = parse_feedback(msg, expected_ids)
            if feedback is None:
                continue
            motor_id, value = feedback
            with self._feedback_lock:
                self._feedback_seq += 1
                self.feedbacks[motor_id] = TimedMotorFeedback(
                    value=value,
                    updated_at=time.monotonic(),
                    seq=self._feedback_seq,
                )
