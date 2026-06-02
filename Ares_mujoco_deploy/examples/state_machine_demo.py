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
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.teleop import VelocityTeleop
from control.config import PositionControlCommand, PositionControlConfig
from sim.ares_mujoco_simulation import AresMuJoCoSimulation, CROUCH_POSE, DT
from sim.ares_state_machine import AresState, AresStateMachine, AresStateMachineConfig


DEFAULT_CONFIG_PATH = ROOT / "config" / "state_machine.yaml"
TELEOP_KEY_LABELS = {
    glfw.KEY_W: "w",
    glfw.KEY_X: "x",
    glfw.KEY_A: "a",
    glfw.KEY_D: "d",
    glfw.KEY_Q: "q",
    glfw.KEY_E: "e",
    glfw.KEY_SPACE: "space",
}


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find state-machine config: {path}")
    with path.open("r") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"State-machine config must be a YAML mapping: {path}")
    return data


def config_value(config: dict, section: str, key: str, default):
    section_data = config.get(section, {})
    if section_data is None:
        return default
    if not isinstance(section_data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return section_data.get(key, default)


def cli_or_config(args: argparse.Namespace, name: str, config: dict, section: str, key: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return config_value(config, section, key, default)


def optional_path(value) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved YAML/CLI config and exit.")
    parser.add_argument("--drop-height", type=float, default=None)
    parser.add_argument("--drop-duration", type=float, default=None)
    parser.add_argument("--stand-duration", type=float, default=None)
    parser.add_argument("--crouch-duration", type=float, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true", default=None)
    parser.add_argument("--show-torque-overlay", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-stand", action="store_true", default=None, help="Automatically request stand after INITIAL.")
    parser.add_argument("--auto-position", action="store_true", default=None, help="Automatically enter position mode after stand.")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run; 0 means forever.")
    parser.add_argument("--position-kp", type=float, default=None)
    parser.add_argument("--position-kd", type=float, default=None)
    parser.add_argument("--step-period", type=float, default=None)
    parser.add_argument("--swing-time", type=float, default=None)
    parser.add_argument("--overlap-time", type=float, default=None)
    parser.add_argument("--z-clearance", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-x-offset", type=float, default=None)
    parser.add_argument("--max-y-offset", type=float, default=None)
    parser.add_argument("--body-forward-bias", type=float, default=None)
    parser.add_argument("--accumulate-foot-targets", action="store_true", default=None)
    parser.add_argument("--command-vx", type=float, default=None, help="Initial forward velocity command in position mode.")
    parser.add_argument("--command-vy", type=float, default=None, help="Initial lateral velocity command in position mode.")
    parser.add_argument("--command-yaw-rate", type=float, default=None, help="Initial yaw-rate command in position mode.")
    parser.add_argument("--yaw-feedback-gain", type=float, default=None)
    parser.add_argument("--max-yaw-rate", type=float, default=None)
    parser.add_argument("--enable-tilt-compensation", action="store_true", default=None)
    parser.add_argument("--tilt-compensation-gain", type=float, default=None)
    parser.add_argument("--max-tilt-compensation", type=float, default=None)
    parser.add_argument("--position-log", type=Path, default=None, help="CSV path for position-mode debug logging.")
    parser.add_argument("--log-interval", type=float, default=None)
    args = parser.parse_args()
    yaml_config = load_config(args.config)

    drop_height = cli_or_config(args, "drop_height", yaml_config, "simulation", "drop_height", 1.0)
    duration = cli_or_config(args, "duration", yaml_config, "simulation", "duration", 0.0)
    no_viewer = cli_or_config(args, "no_viewer", yaml_config, "simulation", "no_viewer", False)
    show_torque_overlay = cli_or_config(
        args,
        "show_torque_overlay",
        yaml_config,
        "display",
        "show_torque_overlay",
        False,
    )
    auto_stand = cli_or_config(args, "auto_stand", yaml_config, "startup", "auto_stand", False)
    auto_position = cli_or_config(args, "auto_position", yaml_config, "startup", "auto_position", False)
    command_vx = cli_or_config(args, "command_vx", yaml_config, "startup", "command_vx", 0.0)
    command_vy = cli_or_config(args, "command_vy", yaml_config, "startup", "command_vy", 0.0)
    command_yaw_rate = cli_or_config(args, "command_yaw_rate", yaml_config, "startup", "command_yaw_rate", 0.0)
    position_log = args.position_log
    if position_log is None:
        position_log = optional_path(config_value(yaml_config, "logging", "position_log", None))
    log_interval = cli_or_config(args, "log_interval", yaml_config, "logging", "log_interval", 0.05)

    teleop = VelocityTeleop()
    commands: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(target=stdin_worker, args=(commands,), daemon=True)
    thread.start()

    position_config = PositionControlConfig(
        kp=cli_or_config(args, "position_kp", yaml_config, "position_control", "kp", 75.0),
        kd=cli_or_config(args, "position_kd", yaml_config, "position_control", "kd", 3.5),
        step_period=cli_or_config(args, "step_period", yaml_config, "position_control", "step_period", 0.34),
        swing_time=cli_or_config(args, "swing_time", yaml_config, "position_control", "swing_time", 0.20),
        overlap_time=cli_or_config(args, "overlap_time", yaml_config, "position_control", "overlap_time", 0.10),
        z_clearance=cli_or_config(args, "z_clearance", yaml_config, "position_control", "z_clearance", 0.07),
        alpha=cli_or_config(args, "alpha", yaml_config, "position_control", "alpha", 3.2),
        beta=cli_or_config(args, "beta", yaml_config, "position_control", "beta", 2.4),
        max_x_offset=cli_or_config(args, "max_x_offset", yaml_config, "position_control", "max_x_offset", 0.32),
        max_y_offset=cli_or_config(args, "max_y_offset", yaml_config, "position_control", "max_y_offset", 0.18),
        body_forward_bias=cli_or_config(args, "body_forward_bias", yaml_config, "position_control", "body_forward_bias", 0.05),
        accumulate_foot_targets=cli_or_config(
            args,
            "accumulate_foot_targets",
            yaml_config,
            "position_control",
            "accumulate_foot_targets",
            False,
        ),
        yaw_feedback_gain=cli_or_config(args, "yaw_feedback_gain", yaml_config, "position_control", "yaw_feedback_gain", 3.0),
        max_yaw_rate=cli_or_config(args, "max_yaw_rate", yaml_config, "position_control", "max_yaw_rate", 2.4),
        enable_tilt_compensation=cli_or_config(
            args,
            "enable_tilt_compensation",
            yaml_config,
            "position_control",
            "enable_tilt_compensation",
            False,
        ),
        tilt_compensation_gain=cli_or_config(
            args,
            "tilt_compensation_gain",
            yaml_config,
            "position_control",
            "tilt_compensation_gain",
            0.8,
        ),
        max_tilt_compensation=cli_or_config(
            args,
            "max_tilt_compensation",
            yaml_config,
            "position_control",
            "max_tilt_compensation",
            0.4,
        ),
    )
    config = AresStateMachineConfig(
        drop_duration=cli_or_config(args, "drop_duration", yaml_config, "state_machine", "drop_duration", 2.0),
        stand_duration=cli_or_config(args, "stand_duration", yaml_config, "state_machine", "stand_duration", 3.0),
        crouch_duration=cli_or_config(args, "crouch_duration", yaml_config, "state_machine", "crouch_duration", 3.0),
        kp=cli_or_config(args, "kp", yaml_config, "state_machine", "kp", 45.0),
        kd=cli_or_config(args, "kd", yaml_config, "state_machine", "kd", 2.0),
        position=position_config,
    )
    if args.print_config:
        resolved_config = {
            "simulation": {
                "drop_height": drop_height,
                "duration": duration,
                "no_viewer": no_viewer,
            },
            "display": {
                "show_torque_overlay": show_torque_overlay,
            },
            "state_machine": {
                "drop_duration": config.drop_duration,
                "stand_duration": config.stand_duration,
                "crouch_duration": config.crouch_duration,
                "kp": config.kp,
                "kd": config.kd,
            },
            "position_control": {
                "kp": position_config.kp,
                "kd": position_config.kd,
                "step_period": position_config.step_period,
                "swing_time": position_config.swing_time,
                "overlap_time": position_config.overlap_time,
                "z_clearance": position_config.z_clearance,
                "alpha": position_config.alpha,
                "beta": position_config.beta,
                "max_x_offset": position_config.max_x_offset,
                "max_y_offset": position_config.max_y_offset,
                "body_forward_bias": position_config.body_forward_bias,
                "accumulate_foot_targets": position_config.accumulate_foot_targets,
                "yaw_feedback_gain": position_config.yaw_feedback_gain,
                "max_yaw_rate": position_config.max_yaw_rate,
                "enable_tilt_compensation": position_config.enable_tilt_compensation,
                "tilt_compensation_gain": position_config.tilt_compensation_gain,
                "max_tilt_compensation": position_config.max_tilt_compensation,
            },
            "startup": {
                "auto_stand": auto_stand,
                "auto_position": auto_position,
                "command_vx": command_vx,
                "command_vy": command_vy,
                "command_yaw_rate": command_yaw_rate,
            },
            "logging": {
                "position_log": None if position_log is None else str(position_log),
                "log_interval": log_interval,
            },
        }
        print(yaml.safe_dump(resolved_config, sort_keys=False), end="")
        return
    if command_vx != 0.0:
        teleop.set_forward_velocity(command_vx)
    if command_vy != 0.0:
        teleop.set_lateral_velocity(command_vy)
    if command_yaw_rate != 0.0:
        teleop.set_yaw_rate(command_yaw_rate)
    if command_vx != 0.0 or command_vy != 0.0 or command_yaw_rate != 0.0:
        print_target_velocity(teleop, "initial command")

    logger = PositionDebugLogger(position_log, log_interval) if position_log is not None else None

    def viewer_key_callback(key: int) -> None:
        teleop.handle_key(key)
        if key in TELEOP_KEY_LABELS:
            print_target_velocity(teleop, f"teleop {TELEOP_KEY_LABELS[key]}")

    try:
        with AresMuJoCoSimulation(
            use_viewer=not no_viewer,
            base_height=drop_height,
            initial_joint_pos=CROUCH_POSE,
            verbose=False,
            key_callback=viewer_key_callback if not no_viewer else None,
            show_torque_overlay=show_torque_overlay,
        ) as sim:
            state_machine = AresStateMachine(sim, config)
            start = time.time()
            last_state = state_machine.state
            auto_stand_sent = False
            auto_position_sent = False
            pending_position = False
            print("状态机已启动：先趴下自由下落，落地后进入 i。")
            print_prompt()

            while sim.is_running() and (duration <= 0.0 or time.time() - start < duration):
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

                if auto_stand and not auto_stand_sent and state_machine.state == AresState.INITIAL:
                    state_machine.request_stand()
                    auto_stand_sent = True
                    print("auto -> s")

                if pending_position and state_machine.state == AresState.STAND:
                    state_machine.request_position(position_command(state_machine))
                    pending_position = False
                    print("-> p")

                if auto_position and not auto_position_sent and state_machine.state == AresState.STAND:
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
