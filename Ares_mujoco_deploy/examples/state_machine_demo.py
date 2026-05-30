from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.teleop import VelocityTeleop
from control.config import PositionControlCommand
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, CROUCH_POSE, DT
from sim.ares_state_machine import AresState, AresStateMachine, AresStateMachineConfig


DROP_HEIGHT = 1.0


def stdin_worker(commands: queue.Queue[str]) -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            commands.put("quit")
            return
        commands.put(line.strip().lower())


def position_command(state_machine: AresStateMachine) -> PositionControlCommand:
    return PositionControlCommand(height=float(state_machine.position_controller.default_stance[2].mean()))


def print_prompt() -> None:
    print("命令：s 站立，p 位置控制，c 趴下，q 退出。位置控制中 WASD 调速度，空格清零。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-duration", type=float, default=2.0)
    parser.add_argument("--stand-duration", type=float, default=3.0)
    parser.add_argument("--kp", type=float, default=45.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--auto-stand", action="store_true", help="Automatically request stand after INITIAL.")
    parser.add_argument("--auto-position", action="store_true", help="Automatically enter position mode after stand.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    args = parser.parse_args()

    teleop = VelocityTeleop()
    commands: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(target=stdin_worker, args=(commands,), daemon=True)
    thread.start()

    config = AresStateMachineConfig(
        drop_duration=args.drop_duration,
        stand_duration=args.stand_duration,
        kp=args.kp,
        kd=args.kd,
    )

    with AresMuJoCoSimulation(
        use_viewer=not args.no_viewer,
        base_height=DROP_HEIGHT,
        initial_joint_pos=CROUCH_POSE,
        verbose=False,
        key_callback=teleop.handle_key if not args.no_viewer else None,
    ) as sim:
        state_machine = AresStateMachine(sim, config)
        start = time.time()
        last_state = state_machine.state
        auto_stand_sent = False
        auto_position_sent = False
        pending_position = False
        print("状态机已启动：先趴下自由下落，落地后进入 i。")
        print_prompt()

        while sim.is_running() and (args.duration <= 0.0 or time.time() - start < args.duration):
            while not commands.empty():
                command = commands.get_nowait()
                if command == "q":
                    return
                if command == "s":
                    if state_machine.request_stand():
                        pending_position = False
                        print("-> s")
                    else:
                        print("当前不能进入 s。")
                elif command == "p":
                    if state_machine.request_position(position_command(state_machine)):
                        print("-> p")
                    elif state_machine.state in (AresState.INITIAL, AresState.CROUCH):
                        if state_machine.request_stand():
                            pending_position = True
                            print("-> s -> p")
                        else:
                            print("当前不能进入 p。")
                    else:
                        print("p 需要先进入 s。")
                elif command == "c":
                    state_machine.request_crouch()
                    pending_position = False
                    teleop.set_zero()
                    print("-> c")
                elif command:
                    print("请输入 c、s、p 或 q。")

            if args.auto_stand and not auto_stand_sent and state_machine.state == AresState.INITIAL:
                state_machine.request_stand()
                auto_stand_sent = True
                print("auto -> s")

            if pending_position and state_machine.state == AresState.STAND:
                state_machine.request_position(position_command(state_machine))
                pending_position = False
                print("-> p")

            if args.auto_position and not auto_position_sent and state_machine.state == AresState.STAND:
                state_machine.request_position(position_command(state_machine))
                auto_position_sent = True
                print("auto -> p")

            if state_machine.state == AresState.POSITION:
                teleop_cmd = teleop.get_command()
                state_machine.set_position_command(
                    PositionControlCommand(
                        horizontal_velocity=teleop_cmd.horizontal_velocity,
                        yaw_rate=teleop_cmd.yaw_rate,
                        height=float(state_machine.position_controller.default_stance[2].mean()),
                    )
                )

            state_machine.step()
            if state_machine.state != last_state:
                last_state = state_machine.state
                if last_state == AresState.INITIAL:
                    print("状态：i")
                elif last_state == AresState.STAND:
                    print("状态：s")
                elif last_state == AresState.POSITION:
                    print("状态：p")
                elif last_state == AresState.CROUCH:
                    print("状态：c")

            time.sleep(DT)


if __name__ == "__main__":
    main()
