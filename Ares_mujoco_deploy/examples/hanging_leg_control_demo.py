from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from control.config import PositionControlCommand
from control.gait import GaitController
from control.position_controller import PositionController
from control.stance_controller import StanceController
from control.swing_controller import SwingController
from control.pinocchio_ik import PinocchioIK
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, DT, STAND_POSE


HANG_HEIGHT = 1.5
FOOT_HEIGHT = -0.30
STEP_PERIOD = 0.42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    with AresMuJoCoSimulation(
        use_viewer=not args.no_viewer,
        base_height=HANG_HEIGHT,
        initial_joint_pos=STAND_POSE,
        verbose=False,
    ) as sim:
        controller = PositionController(sim.model)
        demo_config = replace(
            controller.config,
            overlap_time=0.02,
            swing_time=0.08,
            z_clearance=0.18,
            alpha=1.8,
        )
        gait = GaitController(demo_config)
        stance_controller = StanceController(demo_config)
        swing_controller = SwingController(demo_config, controller.default_stance)
        pin_ik = PinocchioIK()
        controller.reset(sim.get_state()["joint_pos"])
        command = PositionControlCommand(
            horizontal_velocity=np.array([0.14, 0.0], dtype=np.float32),
            height=FOOT_HEIGHT,
        )
        xy_gain = 1.8
        z_gain = 1.8
        start = time.time()
        print("挂起踏步已启动。Ctrl+C 退出。")
        while sim.is_running() and (args.duration <= 0.0 or time.time() - start < args.duration):
            phase = (time.time() - start) % STEP_PERIOD
            step_index = int((phase / STEP_PERIOD) * gait.config.phase_length)
            contacts = gait.contacts(step_index)
            swing_phase = gait.swing_phase(step_index)
            next_locations = controller.foot_locations.copy()
            for leg_index in range(4):
                if contacts[leg_index]:
                    next_locations[:, leg_index] = stance_controller.next_foot_location(
                        leg_index,
                        controller.foot_locations,
                        command,
                    )
                else:
                    next_locations[:, leg_index] = swing_controller.next_foot_location(
                        swing_phase,
                        leg_index,
                        controller.foot_locations,
                        command,
                    )
                base = controller.default_stance[:, leg_index]
                target = next_locations[:, leg_index]
                target = np.array(
                    [
                        np.clip(base[0] + xy_gain * (target[0] - base[0]), base[0] - 0.12, base[0] + 0.12),
                        np.clip(base[1] + xy_gain * (target[1] - base[1]), base[1] - 0.10, base[1] + 0.10),
                        np.clip(base[2] + z_gain * (target[2] - base[2]), base[2] - 0.02, base[2] + 0.16),
                    ],
                    dtype=np.float32,
                )
                next_locations[:, leg_index] = target
            q_des = pin_ik.solve(next_locations, sim.get_state()["joint_pos"], base_pose=np.array([0.0, 0.0, HANG_HEIGHT], dtype=np.float64))
            sim.set_mit_command(
                kp=np.full(12, 75.0, dtype=np.float32),
                q_des=q_des,
                kd=np.full(12, 3.5, dtype=np.float32),
            )
            sim.hold_base_pose(HANG_HEIGHT)
            sim.step()
            time.sleep(DT)


if __name__ == "__main__":
    main()
