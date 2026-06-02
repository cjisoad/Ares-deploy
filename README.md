# Ares_deploy

`Ares_deploy` 是 Ares 四足机器人模型、MuJoCo 仿真和控制实验的工作区。当前主线集中在 `Ares_mujoco_deploy/`：从 `model/Ares` 中的 URDF 和 mesh 生成 MJCF，然后通过 Python 接口运行 MuJoCo、下发 MIT 关节控制命令，并验证站立、趴下、挂起踏步和位置控制等流程。

根目录 README 作为仓库入口。更细的模型姿态表和子模块说明见 `model/Ares/README.md` 与 `Ares_mujoco_deploy/README.md`。

## 当前主线

- `model/Ares/`：Ares 模型源数据，包含 `urdf/Ares.urdf`、mesh、关节名称配置和 ROS/Gazebo 相关模型文件。
- `Ares_mujoco_deploy/assets/Ares.xml`：MuJoCo 运行时直接加载的 MJCF 模型，由转换脚本生成。
- `Ares_mujoco_deploy/sim/`：仿真核心，包含 `AresMuJoCoSimulation` 和上层 `AresStateMachine`。
- `Ares_mujoco_deploy/control/`：控制模块，包含 MIT/PD 参数、遥操作速度命令、步态相位、支撑腿/摆动腿落足点、运动学和 Pinocchio IK。
- `Ares_mujoco_deploy/examples/`：可直接运行的 demo，用于验证模型、关节控制、自由下落、挂起踏步和状态机。
- `Ares_mujoco_deploy/tools/`：模型转换工具，目前主要是 `convert_ares_urdf_to_mjcf.py`。
- `archive/`：历史 Lite3 参考和第三方归档。当前 Ares 主线不依赖这里的代码。

## 环境依赖

建议使用 Python 3.10+。仓库目前没有集中维护 `requirements.txt`，代码实际使用的主要 Python 包如下：

```bash
python3 -m pip install numpy mujoco glfw pin
```

其中 `pin` 提供 Python 里的 `pinocchio` 模块；如果 pip 环境安装不顺，可改用 conda-forge 的 `pinocchio` 包。MuJoCo viewer 需要本机图形环境；如果只想做无界面检查，可给相关脚本加 `--no-viewer`。

## 快速运行

从仓库根目录进入当前仿真主目录：

```bash
cd Ares_mujoco_deploy
```

生成或刷新 MJCF：

```bash
python3 tools/convert_ares_urdf_to_mjcf.py
```

运行基础仿真：

```bash
python3 sim/ares_mujoco_simulation.py
```

常用参数：

```bash
python3 sim/ares_mujoco_simulation.py --base-height 1.0
python3 sim/ares_mujoco_simulation.py --torque-limit 17
python3 sim/ares_mujoco_simulation.py --no-viewer
python3 sim/ares_mujoco_simulation.py --model assets/Ares.xml
```

## 示例脚本

MIT 关节控制接口验证：

```bash
python3 examples/mit_control_demo.py
```

自由下落到蓝色棋盘格地面：

```bash
python3 examples/free_fall_demo.py
```

机身固定在 1.5 m 高度、四肢自然下垂并保持 MIT 控制：

```bash
python3 examples/suspended_mit_demo.py
```

机身固定在 1.5 m 高度，用步态、摆动腿/支撑腿控制和 Pinocchio IK 做挂起踏步：

```bash
python3 examples/hanging_leg_control_demo.py
```

状态机 demo：

```bash
python3 examples/state_machine_demo.py
```

状态机启动后先以趴下姿态自由下落，之后进入初始状态。终端命令如下：

- `s`：从初始/趴下/站立相关状态进入站立过渡。
- `p`：进入位置控制；若还未站立，会先请求站立再进入位置控制。
- `c`：从当前状态进入趴下过渡。
- `w` / `x`：位置控制中调整前进/后退速度。
- `a` / `d`：位置控制中调整横向速度。
- `q` / `e`：位置控制中调整 yaw rate。
- `space`、`clear` 或 `0`：清零遥操作命令。
- `quit` 或 `exit`：退出。

常用自动化参数：

```bash
python3 examples/state_machine_demo.py --auto-stand
python3 examples/state_machine_demo.py --auto-stand --auto-position
python3 examples/state_machine_demo.py --duration 10 --no-viewer
```

## Python 接口

核心仿真类是 `AresMuJoCoSimulation`。它默认加载 `Ares_mujoco_deploy/assets/Ares.xml`，控制 12 个关节，仿真步长为 `0.001 s`，默认力矩限幅为 `17 N*m`。

```python
import numpy as np

from sim.ares_mujoco_simulation import AresMuJoCoSimulation, DEFAULT_STAND

with AresMuJoCoSimulation(use_viewer=False) as sim:
    sim.set_mit_command(
        kp=np.full(12, 80.0, dtype=np.float32),
        q_des=DEFAULT_STAND,
        kd=np.full(12, 2.0, dtype=np.float32),
    )
    state = sim.step()
```

MIT 控制律：

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```

`step()` 返回的状态字段包括：

- `time`：仿真时间。
- `base_rpy`：机身 roll、pitch、yaw。
- `base_omega`：IMU 角速度传感器输出。
- `base_acc`：IMU 加速度传感器输出。
- `joint_pos`：12 关节位置。
- `joint_vel`：12 关节速度。
- `joint_tau`：实际下发到 MuJoCo actuator 的限幅后力矩。

## 模型与控制维护

- 修改几何、惯量、关节限位或 mesh 时，优先维护 `model/Ares/`，然后运行 `Ares_mujoco_deploy/tools/convert_ares_urdf_to_mjcf.py` 刷新 `assets/Ares.xml`。
- 站立和趴下姿态在转换脚本、仿真初始姿态和 `model/Ares/README.md` 中都有对应数据；调整时需要保持这些入口一致。
- `JOINT_ORDER` 固定为 `lf`、`rf`、`lb`、`rb`，每条腿按 `hip_base`、`thigh_hip`、`calf_thigh` 排列。控制命令、状态返回和姿态数组都按这个顺序解释。
- 当前主线代码在 `Ares_mujoco_deploy/` 内运行。除非明确整理历史参考，不建议把 `archive/` 作为功能修改目标。

## 常见问题

- `ModuleNotFoundError: No module named 'pinocchio'`：安装 `pin` 包，或确认当前 Python 环境可导入 `pinocchio`。
- viewer 无法打开：先用 `--no-viewer` 跑无界面流程，确认 MuJoCo 和模型加载正常；图形界面问题通常与本机 OpenGL/显示环境有关。
- 修改 URDF 后仿真没有变化：需要重新运行 `python3 tools/convert_ares_urdf_to_mjcf.py`，仿真加载的是 `assets/Ares.xml`。
