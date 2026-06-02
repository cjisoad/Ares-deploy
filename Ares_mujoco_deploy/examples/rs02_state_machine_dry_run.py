from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_io.rs02.backend import RS02HardwareBackend
from robot_io.rs02.config import load_rs02_config
from sim.ares_state_machine import AresState, AresStateMachine, AresStateMachineConfig


DEFAULT_CONFIG_PATH = ROOT / "config" / "rs02_hardware.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ares state machine on the RS02 dry-run backend")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--auto-stand", action="store_true")
    parser.add_argument("--auto-position", action="store_true")
    parser.add_argument("--kp", type=float, default=45.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--position-kp", type=float, default=30.0)
    parser.add_argument("--position-kd", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = RS02HardwareBackend(load_rs02_config(args.config), live=False)
    config = AresStateMachineConfig(kp=args.kp, kd=args.kd)
    config.position.kp = args.position_kp
    config.position.kd = args.position_kd
    backend.start()
    try:
        state_machine = AresStateMachine(backend, config)
        start = time.monotonic()
        last_state = state_machine.state
        auto_stand_sent = False
        auto_position_sent = False
        last_print = start
        while backend.is_running() and time.monotonic() - start < args.duration:
            if args.auto_stand and not auto_stand_sent and state_machine.state == AresState.INITIAL:
                state_machine.request_stand()
                auto_stand_sent = True
            if args.auto_position and not auto_position_sent and state_machine.state == AresState.STAND:
                state_machine.request_position()
                auto_position_sent = True
            state = state_machine.step()
            if state_machine.state != last_state:
                last_state = state_machine.state
                print(f"state={last_state.value}")
            now = time.monotonic()
            if now - last_print >= 0.5:
                last_print = now
                joint_pos = np.asarray(state["joint_pos"], dtype=np.float32)
                print(f"t={state['time']:.3f} state={state_machine.state.value} q0={joint_pos[0]:+.3f}")
            time.sleep(backend.dt)
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
