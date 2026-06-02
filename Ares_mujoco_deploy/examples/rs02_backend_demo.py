from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_io.rs02.backend import RS02HardwareBackend
from robot_io.rs02.config import load_rs02_config


DEFAULT_CONFIG_PATH = ROOT / "config" / "rs02_hardware.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RS02 12-joint backend smoke test")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--kp", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument("--live", action="store_true", help="send commands to real RS02 motors")
    parser.add_argument(
        "--i-understand-live-rs02",
        action="store_true",
        help="required together with --live",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_rs02_config(args.config)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False), end="")
        return
    if args.live and not args.i_understand_live_rs02:
        raise RuntimeError("Live RS02 output requires --i-understand-live-rs02")

    backend = RS02HardwareBackend(config, live=args.live)
    q_des = np.zeros(backend.dof_num, dtype=np.float32)
    kp = np.full(backend.dof_num, args.kp, dtype=np.float32)
    kd = np.full(backend.dof_num, args.kd, dtype=np.float32)
    dq_des = np.zeros(backend.dof_num, dtype=np.float32)
    tau_ff = np.zeros(backend.dof_num, dtype=np.float32)

    backend.start()
    try:
        start = time.monotonic()
        last_print = start
        while backend.is_running() and time.monotonic() - start < args.duration:
            backend.set_mit_command(kp=kp, q_des=q_des, kd=kd, dq_des=dq_des, tau_ff=tau_ff)
            state = backend.step()
            now = time.monotonic()
            if now - last_print >= 0.5:
                last_print = now
                joint_pos = np.asarray(state["joint_pos"], dtype=np.float32)
                joint_tau = np.asarray(state["joint_tau"], dtype=np.float32)
                print(
                    f"t={state['time']:.3f} "
                    f"q0={joint_pos[0]:+.3f} "
                    f"tau_abs_max={float(np.max(np.abs(joint_tau))):.3f}"
                )
            time.sleep(backend.dt)
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
