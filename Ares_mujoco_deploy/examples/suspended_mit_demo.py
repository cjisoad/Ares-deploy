from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.ares_mujoco_simulation import AresMuJoCoSimulation, DT


HANG_HEIGHT = 1.5
NATURAL_HANG = np.zeros(12, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    sim = AresMuJoCoSimulation(
        use_viewer=not args.no_viewer,
        base_height=HANG_HEIGHT,
        initial_joint_pos=NATURAL_HANG,
    )

    kp = np.full(12, 30.0, dtype=np.float32)
    kd = np.full(12, 1.5, dtype=np.float32)
    dq_des = np.zeros(12, dtype=np.float32)
    tau_ff = np.zeros(12, dtype=np.float32)

    start = time.time()
    while args.duration <= 0.0 or time.time() - start < args.duration:
        sim.hold_base_pose(HANG_HEIGHT)
        sim.set_mit_command(
            kp=kp,
            q_des=NATURAL_HANG,
            kd=kd,
            dq_des=dq_des,
            tau_ff=tau_ff,
        )
        sim.step()
        sim.hold_base_pose(HANG_HEIGHT)
        time.sleep(DT)


if __name__ == "__main__":
    main()
