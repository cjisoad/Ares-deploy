from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.config import PositionControlCommand
from control.position_controller import PositionController
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, DT, STAND_POSE


HANG_HEIGHT = 1.5
FOOT_HEIGHT = -0.30
STEP_PERIOD = 0.8
STEP_LIFT = 0.45
STEP_KNEE = 0.35
WALK_STEP_HEIGHT = 0.08


def build_step_pattern(controller: PositionController, command: PositionControlCommand, phase: float) -> np.ndarray:
    foot = controller.default_stance.copy()
    stride = 0.12 * command.horizontal_velocity
    local_phase = (phase * 2.0) % 1.0
    lift = 0.07 * np.sin(np.pi * local_phase)
    if phase < 0.5:
        swing_legs = (0, 3)
        support_legs = (1, 2)
    else:
        swing_legs = (1, 2)
        support_legs = (0, 3)
    for leg in support_legs:
        foot[0, leg] -= stride[0] * 0.5
        foot[1, leg] -= stride[1] * 0.5
        foot[2, leg] = command.height
    for leg in swing_legs:
        foot[0, leg] += stride[0] * (local_phase - 0.5)
        foot[1, leg] += stride[1] * (local_phase - 0.5)
        foot[2, leg] = command.height + max(lift, 0.0)
    return foot


def joint_space_step(phase: float) -> np.ndarray:
    q = STAND_POSE.copy().reshape(4, 3)
    local_phase = (phase * 2.0) % 1.0
    lift = np.sin(np.pi * local_phase)
    if phase < 0.5:
        swing_legs = (0, 3)
    else:
        swing_legs = (1, 2)
    for leg in swing_legs:
        thigh_sign = -1.0 if leg in (0, 2) else 1.0
        calf_sign = 1.0 if leg in (0, 2) else -1.0
        q[leg, 1] += thigh_sign * STEP_LIFT * lift
        q[leg, 2] += calf_sign * STEP_KNEE * lift
    return q.reshape(12).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--mode", choices=("joint", "ik"), default="joint", help="joint is easier to see; ik uses foot targets.")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    with AresMuJoCoSimulation(
        use_viewer=not args.no_viewer,
        base_height=HANG_HEIGHT,
        initial_joint_pos=STAND_POSE,
        verbose=False,
    ) as sim:
        controller = PositionController(sim.model)
        controller.reset(sim.get_state()["joint_pos"])
        phase = 0.0
        command = PositionControlCommand(horizontal_velocity=np.array([0.10, 0.0], dtype=np.float32), height=FOOT_HEIGHT)
        start = time.time()
        print(f"挂起踏步已启动，mode={args.mode}。Ctrl+C 退出。")
        while sim.is_running() and (args.duration <= 0.0 or time.time() - start < args.duration):
            phase = (phase + DT / STEP_PERIOD) % 1.0
            if args.mode == "joint":
                q_des = joint_space_step(phase)
            else:
                target_feet = build_step_pattern(controller, command, phase)
                q_des = controller.set_foot_pattern(target_feet, sim.get_state()["joint_pos"])
            sim.set_mit_command(
                kp=np.full(12, 60.0, dtype=np.float32),
                q_des=q_des,
                kd=np.full(12, 3.0, dtype=np.float32),
            )
            sim.hold_base_pose(HANG_HEIGHT)
            sim.step()
            time.sleep(DT)


if __name__ == "__main__":
    main()
