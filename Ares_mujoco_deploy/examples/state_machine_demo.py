from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import glfw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.teleop import VelocityTeleop
from control.config import PositionControlCommand, PositionControlConfig
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, CROUCH_POSE, DT
from sim.ares_state_machine import AresState, AresStateMachine, AresStateMachineConfig


DROP_HEIGHT = 1.0
TELEOP_KEY_LABELS = {
    glfw.KEY_W: "w",
    glfw.KEY_X: "x",
    glfw.KEY_A: "a",
    glfw.KEY_D: "d",
    glfw.KEY_Q: "q",
    glfw.KEY_E: "e",
    glfw.KEY_SPACE: "space",
}


def parse_stdin_command(line: str) -> str:
    command = line.strip().lower()
    if command:
        return command
    if line != "\n" and any(ch.isspace() for ch in line):
        return "space"
    return ""


def stdin_worker(commands: queue.Queue[str]) -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        command = parse_stdin_command(line)
        if command:
            commands.put(command)


def position_command(state_machine: AresStateMachine) -> PositionControlCommand:
    return PositionControlCommand(height=float(state_machine.position_controller.default_stance[2].mean()))


def print_prompt() -> None:
    print("命令：s 站立，p 位置控制，c 趴下，quit 退出。p 中 W/X 前后，A/D 左右，Q/E 转向，空格清零。")


def target_velocity_text(teleop: VelocityTeleop) -> str:
    command = teleop.get_command()
    return (
        f"目标速度 vx={command.horizontal_velocity[0]:+.2f} m/s, "
        f"vy={command.horizontal_velocity[1]:+.2f} m/s, "
        f"yaw_rate={command.yaw_rate:+.2f} rad/s"
    )


def print_target_velocity(teleop: VelocityTeleop, label: str) -> None:
    print(f"{label} -> {target_velocity_text(teleop)}")


class PositionDebugLogger:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.last_log_time = -interval
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "time",
                "base_x",
                "base_y",
                "base_z",
                "roll",
                "pitch",
                "yaw",
                "cmd_vx",
                "cmd_vy",
                "cmd_yaw_rate",
                "effective_yaw_rate",
                "tau_abs_max",
                "tau_saturation_ratio",
                "target_q0",
                "target_q1",
                "target_q2",
                "contact_lf",
                "contact_rf",
                "contact_lb",
                "contact_rb",
                "foot_lf_x",
                "foot_rf_x",
                "foot_lb_x",
                "foot_rb_x",
            ]
        )

    def close(self) -> None:
        self.file.close()

    def maybe_write(self, state_machine: AresStateMachine, state: dict[str, np.ndarray | float]) -> None:
        timestamp = float(state["time"])
        if timestamp - self.last_log_time < self.interval:
            return
        self.last_log_time = timestamp
        sim = state_machine.sim
        command = state_machine.position_command
        base_pos = np.asarray(state["base_pos"], dtype=np.float32)
        base_rpy = np.asarray(state["base_rpy"], dtype=np.float32)
        tau = np.asarray(state["joint_tau"], dtype=np.float32)
        target = np.asarray(state_machine.last_position_target, dtype=np.float32)
        contacts = np.asarray(state_machine.last_position_contacts, dtype=np.int32)
        foot_targets = np.asarray(state_machine.position_controller.last_foot_targets, dtype=np.float32)
        tau_abs = np.abs(tau)
        saturation_ratio = float(np.mean(tau_abs >= 0.98 * sim.torque_limit))
        self.writer.writerow(
            [
                f"{timestamp:.4f}",
                f"{base_pos[0]:.6f}",
                f"{base_pos[1]:.6f}",
                f"{base_pos[2]:.6f}",
                f"{base_rpy[0]:.6f}",
                f"{base_rpy[1]:.6f}",
                f"{base_rpy[2]:.6f}",
                f"{command.horizontal_velocity[0]:.6f}",
                f"{command.horizontal_velocity[1]:.6f}",
                f"{command.yaw_rate:.6f}",
                f"{state_machine.last_effective_yaw_rate:.6f}",
                f"{float(np.max(tau_abs)):.6f}",
                f"{saturation_ratio:.6f}",
                f"{target[0]:.6f}",
                f"{target[1]:.6f}",
                f"{target[2]:.6f}",
                int(contacts[0]),
                int(contacts[1]),
                int(contacts[2]),
                int(contacts[3]),
                f"{foot_targets[0, 0]:.6f}",
                f"{foot_targets[0, 1]:.6f}",
                f"{foot_targets[0, 2]:.6f}",
                f"{foot_targets[0, 3]:.6f}",
            ]
        )
        self.file.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-duration", type=float, default=2.0)
    parser.add_argument("--stand-duration", type=float, default=3.0)
    parser.add_argument("--crouch-duration", type=float, default=3.0)
    parser.add_argument("--kp", type=float, default=45.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--auto-stand", action="store_true", help="Automatically request stand after INITIAL.")
    parser.add_argument("--auto-position", action="store_true", help="Automatically enter position mode after stand.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means forever.")
    parser.add_argument("--position-kp", type=float, default=75.0)
    parser.add_argument("--position-kd", type=float, default=3.5)
    parser.add_argument("--step-period", type=float, default=0.34)
    parser.add_argument("--swing-time", type=float, default=0.20)
    parser.add_argument("--overlap-time", type=float, default=0.10)
    parser.add_argument("--z-clearance", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=3.2)
    parser.add_argument("--beta", type=float, default=2.4)
    parser.add_argument("--max-x-offset", type=float, default=0.32)
    parser.add_argument("--max-y-offset", type=float, default=0.18)
    parser.add_argument("--body-forward-bias", type=float, default=0.05)
    parser.add_argument("--accumulate-foot-targets", action="store_true")
    parser.add_argument("--command-vx", type=float, default=0.0, help="Initial forward velocity command in position mode.")
    parser.add_argument("--command-vy", type=float, default=0.0, help="Initial lateral velocity command in position mode.")
    parser.add_argument("--command-yaw-rate", type=float, default=0.0, help="Initial yaw-rate command in position mode.")
    parser.add_argument("--yaw-feedback-gain", type=float, default=3.0)
    parser.add_argument("--max-yaw-rate", type=float, default=2.4)
    parser.add_argument("--enable-tilt-compensation", action="store_true")
    parser.add_argument("--tilt-compensation-gain", type=float, default=0.8)
    parser.add_argument("--max-tilt-compensation", type=float, default=0.4)
    parser.add_argument("--position-log", type=Path, default=None, help="CSV path for position-mode debug logging.")
    parser.add_argument("--log-interval", type=float, default=0.05)
    args = parser.parse_args()

    teleop = VelocityTeleop()
    commands: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(target=stdin_worker, args=(commands,), daemon=True)
    thread.start()

    position_config = PositionControlConfig(
        kp=args.position_kp,
        kd=args.position_kd,
        step_period=args.step_period,
        swing_time=args.swing_time,
        overlap_time=args.overlap_time,
        z_clearance=args.z_clearance,
        alpha=args.alpha,
        beta=args.beta,
        max_x_offset=args.max_x_offset,
        max_y_offset=args.max_y_offset,
        body_forward_bias=args.body_forward_bias,
        accumulate_foot_targets=args.accumulate_foot_targets,
        yaw_feedback_gain=args.yaw_feedback_gain,
        max_yaw_rate=args.max_yaw_rate,
        enable_tilt_compensation=args.enable_tilt_compensation,
        tilt_compensation_gain=args.tilt_compensation_gain,
        max_tilt_compensation=args.max_tilt_compensation,
    )
    config = AresStateMachineConfig(
        drop_duration=args.drop_duration,
        stand_duration=args.stand_duration,
        crouch_duration=args.crouch_duration,
        kp=args.kp,
        kd=args.kd,
        position=position_config,
    )
    if args.command_vx != 0.0:
        teleop.set_forward_velocity(args.command_vx)
    if args.command_vy != 0.0:
        teleop.set_lateral_velocity(args.command_vy)
    if args.command_yaw_rate != 0.0:
        teleop.set_yaw_rate(args.command_yaw_rate)
    if args.command_vx != 0.0 or args.command_vy != 0.0 or args.command_yaw_rate != 0.0:
        print_target_velocity(teleop, "initial command")

    logger = PositionDebugLogger(args.position_log, args.log_interval) if args.position_log is not None else None

    def viewer_key_callback(key: int) -> None:
        teleop.handle_key(key)
        if key in TELEOP_KEY_LABELS:
            print_target_velocity(teleop, f"teleop {TELEOP_KEY_LABELS[key]}")

    try:
        with AresMuJoCoSimulation(
            use_viewer=not args.no_viewer,
            base_height=DROP_HEIGHT,
            initial_joint_pos=CROUCH_POSE,
            verbose=False,
            key_callback=viewer_key_callback if not args.no_viewer else None,
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
                    if command in ("quit", "exit"):
                        return
                    if command == "w":
                        teleop.handle_key(glfw.KEY_W)
                        print_target_velocity(teleop, "teleop w")
                        continue
                    if command == "x":
                        teleop.handle_key(glfw.KEY_X)
                        print_target_velocity(teleop, "teleop x")
                        continue
                    if command == "a":
                        teleop.handle_key(glfw.KEY_A)
                        print_target_velocity(teleop, "teleop a")
                        continue
                    if command == "d":
                        teleop.handle_key(glfw.KEY_D)
                        print_target_velocity(teleop, "teleop d")
                        continue
                    if command == "q":
                        teleop.handle_key(glfw.KEY_Q)
                        print_target_velocity(teleop, "teleop q")
                        continue
                    if command == "e":
                        teleop.handle_key(glfw.KEY_E)
                        print_target_velocity(teleop, "teleop e")
                        continue
                    if command in ("space", "clear", "0"):
                        teleop.set_zero()
                        print_target_velocity(teleop, "teleop clear")
                        continue
                    if command == "s":
                        if state_machine.request_stand():
                            pending_position = False
                            teleop.set_zero()
                            print("-> s")
                            print_target_velocity(teleop, "teleop clear")
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
                        print_target_velocity(teleop, "teleop clear")
                    elif command:
                        print("请输入 s、p、c、quit，或 p 状态下的 w/x/a/d/q/e/space。")

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

                state = state_machine.step()
                if logger is not None and state_machine.state == AresState.POSITION:
                    logger.maybe_write(state_machine, state)
                if state_machine.state != last_state:
                    last_state = state_machine.state
                    if last_state == AresState.INITIAL:
                        print("状态：i")
                    elif last_state == AresState.STAND:
                        print("状态：s")
                    elif last_state == AresState.POSITION:
                        print("状态：p")
                    elif last_state == AresState.CROUCHING:
                        print("状态：c...")
                    elif last_state == AresState.CROUCH:
                        print("状态：c")

                time.sleep(DT)
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
