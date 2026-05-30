from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.ares_mujoco_simulation import AresMuJoCoSimulation, DEFAULT_STAND, DT


def main() -> None:
    with AresMuJoCoSimulation(use_viewer=True, base_height=0.8) as sim:
        kp = np.full(12, 80.0, dtype=np.float32)
        kd = np.full(12, 2.0, dtype=np.float32)
        q_des = DEFAULT_STAND.copy()
        dq_des = np.zeros(12, dtype=np.float32)
        tau_ff = np.zeros(12, dtype=np.float32)

        start = time.time()
        while sim.is_running():
            t = time.time() - start

            q_cmd = q_des.copy()
            q_cmd[[0, 3, 6, 9]] += 0.15 * np.sin(2.0 * np.pi * 0.5 * t)

            sim.set_mit_command(
                kp=kp,
                q_des=q_cmd,
                kd=kd,
                dq_des=dq_des,
                tau_ff=tau_ff,
            )
            state = sim.step()

            if int(state["time"] * 10) != int((state["time"] - DT) * 10):
                print(
                    f"t={state['time']:.2f} "
                    f"q0={state['joint_pos'][0]:.3f} "
                    f"tau0={state['joint_tau'][0]:.3f}"
                )

            time.sleep(DT)


if __name__ == "__main__":
    main()
