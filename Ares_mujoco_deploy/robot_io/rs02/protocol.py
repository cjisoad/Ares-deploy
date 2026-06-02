from __future__ import annotations

from dataclasses import dataclass
from typing import Any


P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -44.0, 44.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
T_MIN, T_MAX = -17.0, 17.0

COMM_MOTION_CONTROL = 0x01
COMM_MOTOR_FEEDBACK = 0x02
COMM_MOTOR_ENABLE = 0x03
COMM_MOTOR_STOP = 0x04
COMM_SET_SINGLE_PARAMETER = 0x12

RUN_MODE = 0x7005
MIT_RUN_MODE = 0


@dataclass(frozen=True)
class MotorFeedback:
    angle: float
    speed: float
    torque: float
    temp: float
    error_code: int
    pattern: int


@dataclass(frozen=True)
class TimedMotorFeedback:
    value: MotorFeedback
    updated_at: float
    seq: int


@dataclass(frozen=True)
class RS02MitCommand:
    torque: float = 0.0
    angle: float = 0.0
    speed: float = 0.0
    kp: float = 0.0
    kd: float = 0.0

    def clamped(self) -> "RS02MitCommand":
        return RS02MitCommand(
            torque=clamp(self.torque, T_MIN, T_MAX),
            angle=clamp(self.angle, P_MIN, P_MAX),
            speed=clamp(self.speed, V_MIN, V_MAX),
            kp=clamp(self.kp, KP_MIN, KP_MAX),
            kd=clamp(self.kd, KD_MIN, KD_MAX),
        )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def float_to_uint(value: float, low: float, high: float, bits: int = 16) -> int:
    value = clamp(value, low, high)
    span = high - low
    return int((value - low) * ((1 << bits) - 1) / span)


def uint_to_float(value: int, low: float, high: float, bits: int = 16) -> float:
    span = (1 << bits) - 1
    return (high - low) * value / span + low


def parse_feedback(msg: Any, expected_motor_ids: set[int]) -> tuple[int, MotorFeedback] | None:
    arbitration_id = int(msg.arbitration_id)
    comm_type = (arbitration_id >> 24) & 0x3F
    source_motor_id = (arbitration_id >> 8) & 0xFF
    if comm_type != COMM_MOTOR_FEEDBACK or source_motor_id not in expected_motor_ids:
        return None

    data = bytes(msg.data)
    if len(data) < 8:
        return None

    return (
        source_motor_id,
        MotorFeedback(
            angle=uint_to_float((data[0] << 8) | data[1], P_MIN, P_MAX),
            speed=uint_to_float((data[2] << 8) | data[3], V_MIN, V_MAX),
            torque=uint_to_float((data[4] << 8) | data[5], T_MIN, T_MAX),
            temp=((data[6] << 8) | data[7]) * 0.1,
            error_code=(arbitration_id >> 16) & 0x3F,
            pattern=(arbitration_id >> 22) & 0x03,
        ),
    )
