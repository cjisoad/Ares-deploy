from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from sim.ares_mujoco_simulation import AresMuJoCoSimulation, CROUCH_POSE, DT


DROP_HEIGHT = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    with AresMuJoCoSimulation(
        use_viewer=not args.no_viewer,
        base_height=DROP_HEIGHT,
        initial_joint_pos=CROUCH_POSE,
    ) as sim:
        sim.set_mit_command(
            kp=np.zeros(12, dtype=np.float32),
            q_des=CROUCH_POSE,
            kd=np.zeros(12, dtype=np.float32),
        )

        start = time.time()
        while sim.is_running() and (args.duration <= 0.0 or time.time() - start < args.duration):
            sim.step()
            time.sleep(DT)


if __name__ == "__main__":
    main()
