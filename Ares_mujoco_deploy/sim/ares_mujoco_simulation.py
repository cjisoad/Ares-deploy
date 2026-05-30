from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DT = 0.001
DEFAULT_BASE_HEIGHT = 0.8
DEFAULT_TORQUE_LIMIT = 17.0

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "Ares.xml"
JOINT_ORDER = [
    "lf_hip_base_joint",
    "lf_thigh_hip_joint",
    "lf_calf_thigh_joint",
    "rf_hip_base_joint",
    "rf_thigh_hip_joint",
    "rf_calf_thigh_joint",
    "lb_hip_base_joint",
    "lb_thigh_hip_joint",
    "lb_calf_thigh_joint",
    "rb_hip_base_joint",
    "rb_thigh_hip_joint",
    "rb_calf_thigh_joint",
]
DEFAULT_STAND = np.array([0.0, -1.35453, 2.54948] * 4, dtype=np.float32)


class AresMuJoCoSimulation:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        use_viewer: bool = True,
        base_height: float = DEFAULT_BASE_HEIGHT,
        torque_limit: float = DEFAULT_TORQUE_LIMIT,
    ) -> None:
        self.base_height = base_height
        self.use_viewer = use_viewer
        self.torque_limit = torque_limit

        if not model_path.is_file():
            raise FileNotFoundError(f"Cannot find MJCF model: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        self.dof_num = 12
        self.kp_cmd = np.full((self.dof_num, 1), 80.0, dtype=np.float32)
        self.kd_cmd = np.full((self.dof_num, 1), 2.0, dtype=np.float32)
        self.pos_cmd = DEFAULT_STAND.reshape(self.dof_num, 1).copy()
        self.vel_cmd = np.zeros((self.dof_num, 1), dtype=np.float32)
        self.tau_ff = np.zeros((self.dof_num, 1), dtype=np.float32)
        self.input_tq = np.zeros((self.dof_num, 1), dtype=np.float32)
        self.timestamp = 0.0
        self.last_print = 0.0
        self.step_count = 0

        self._set_initial_pose()

        self.viewer = None
        if use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        print(f"[INFO] Ares MuJoCo model loaded from {model_path}")
        print("[INFO] Python MIT joint-control interface enabled")

    def _set_initial_pose(self) -> None:
        qpos = self.data.qpos.copy()
        qpos[2] = self.base_height
        qpos[7:19] = DEFAULT_STAND
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _joint_pos(self) -> np.ndarray:
        return self.data.qpos[7:19].astype(np.float32)

    def _joint_vel(self) -> np.ndarray:
        return self.data.qvel[6:18].astype(np.float32)

    def set_mit_command(
        self,
        kp: np.ndarray,
        q_des: np.ndarray,
        kd: np.ndarray,
        dq_des: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None:
        """Set 12-joint MIT command: tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff."""
        self.kp_cmd = np.asarray(kp, dtype=np.float32).reshape(self.dof_num, 1)
        self.pos_cmd = np.asarray(q_des, dtype=np.float32).reshape(self.dof_num, 1)
        self.kd_cmd = np.asarray(kd, dtype=np.float32).reshape(self.dof_num, 1)
        if dq_des is None:
            self.vel_cmd = np.zeros((self.dof_num, 1), dtype=np.float32)
        else:
            self.vel_cmd = np.asarray(dq_des, dtype=np.float32).reshape(self.dof_num, 1)
        if tau_ff is None:
            self.tau_ff = np.zeros((self.dof_num, 1), dtype=np.float32)
        else:
            self.tau_ff = np.asarray(tau_ff, dtype=np.float32).reshape(self.dof_num, 1)

    def get_state(self) -> dict[str, np.ndarray | float]:
        return {
            "time": float(self.timestamp),
            "base_rpy": self._quat_to_rpy(self.data.qpos[3:7]),
            "base_omega": self.data.sensordata[0:3].astype(np.float32),
            "base_acc": self.data.sensordata[3:6].astype(np.float32),
            "joint_pos": self._joint_pos(),
            "joint_vel": self._joint_vel(),
            "joint_tau": self.input_tq.flatten().astype(np.float32),
        }

    def _apply_joint_torque(self) -> None:
        q = self._joint_pos().reshape(-1, 1)
        dq = self._joint_vel().reshape(-1, 1)
        self.input_tq = self.kp_cmd * (self.pos_cmd - q) + self.kd_cmd * (self.vel_cmd - dq) + self.tau_ff
        self.input_tq = np.clip(self.input_tq, -self.torque_limit, self.torque_limit)
        self.data.ctrl[:] = self.input_tq.flatten()

    def _debug_print(self) -> None:
        if time.perf_counter() - self.last_print < 2.0:
            return
        self.last_print = time.perf_counter()
        print(f"[Ares] t={self.timestamp:.3f} q0={self._joint_pos()[0]:.3f} tau0={self.input_tq.flatten()[0]:.3f}")

    def step(self) -> dict[str, np.ndarray | float]:
        self._apply_joint_torque()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.timestamp = self.step_count * DT
        if self.viewer is not None and self.step_count % 10 == 0:
            self.viewer.sync()
        return self.get_state()

    def run(self) -> None:
        last_time = time.time()
        while True:
            if time.time() - last_time < DT:
                time.sleep(0.0001)
                continue
            last_time = time.time()
            self.step()
            self._debug_print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--base-height", type=float, default=DEFAULT_BASE_HEIGHT)
    parser.add_argument("--torque-limit", type=float, default=DEFAULT_TORQUE_LIMIT)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    sim = AresMuJoCoSimulation(
        model_path=args.model,
        use_viewer=not args.no_viewer,
        base_height=args.base_height,
        torque_limit=args.torque_limit,
    )
    sim.run()


if __name__ == "__main__":
    main()
