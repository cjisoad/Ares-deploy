# Ares MuJoCo 仿真仓库

这是一个面向 Ares 机型的 MuJoCo Python 仿真仓库，当前功能链路只围绕 Ares 本体展开，不依赖旧的 Lite3 参考实现。

## 仓库结构

当前仓库的运行主链路是 `tools -> assets -> sim -> examples`。

- `tools/`：模型转换脚本目录。
  - `convert_ares_urdf_to_mjcf.py` 会把 Ares 的 URDF 和网格资源转换成 MuJoCo 可加载的 MJCF 文件。
  - 该脚本的输出写入 `assets/Ares.xml`。
- `assets/`：运行时模型资源目录。
  - 目前主要放置 `Ares.xml`，它是仿真器直接加载的 MJCF 模型文件。
- `sim/`：仿真核心目录。
  - `ares_mujoco_simulation.py` 定义 `AresMuJoCoSimulation`，负责加载模型、初始化姿态、执行步进、输出状态，并实现 MIT 风格的关节力矩控制接口。
- `robot_io/`：统一机器人底层 I/O 接口目录。
  - `types.py` 定义 `RobotBackend`、`MitJointCommand`、`RobotState`。
  - `sim_backend.py` 把 MuJoCo 仿真适配成统一 backend。
  - `rs02/` 独立放置 RS02 MIT 电机协议、串口总线、发送/接收线程和 12 关节硬件 backend。
- `examples/`：示例目录。
  - `mit_control_demo.py` 演示如何直接调用 Python API 下发 MIT 控制指令并运行仿真。
  - `suspended_mit_demo.py` 演示机身固定在离地 1.5 米、四肢自然下垂并使用 MIT 力矩控制的场景。
  - `hanging_leg_control_demo.py` 演示机身固定在离地 1.5 米时，用 IK 驱动四条腿做最小摆动和支撑腿补偿，专门用于验证腿部逆解与分腿控制。
  - `free_fall_demo.py` 演示机身从离地 1 米自然下落至蓝色棋盘格地面的场景。
  - `state_machine_demo.py` 演示“初始状态 -> 站立”的上层状态机流程，支持终端手动触发站立。
    进入位置控制后可用 `WASD` 调速度，`Space` 清零。
- `archive/`：归档参考目录。
  - 如果仓库中存在该目录，它只保留历史参考文件，不属于当前功能代码，不参与运行、导入或构建。

## 主要用法

### 生成 MJCF

```bash
cd /home/eden/Ares_deploy/Ares_mujoco_deploy
python3 tools/convert_ares_urdf_to_mjcf.py
```

### 运行仿真

```bash
cd /home/eden/Ares_deploy/Ares_mujoco_deploy
python3 sim/ares_mujoco_simulation.py
```

仿真默认从悬空站立姿态启动。可以通过参数调整底盘初始高度：

```bash
python3 sim/ares_mujoco_simulation.py --base-height 1.0
```

无界面运行：

```bash
python3 sim/ares_mujoco_simulation.py --no-viewer
```

### MIT 控制示例

```bash
cd /home/eden/Ares_deploy/Ares_mujoco_deploy
python3 examples/mit_control_demo.py
```

### 指定场景示例

```bash
python3 examples/suspended_mit_demo.py
python3 examples/hanging_leg_control_demo.py
python3 examples/free_fall_demo.py
```

### 状态机参数配置

状态机 demo 默认读取 `config/state_machine.yaml`。可以直接修改该文件里的站立/趴下时长、`kp/kd`、位置控制参数、自动启动、日志参数，以及 MuJoCo viewer 右上角的关节力矩显示开关：

```bash
python3 examples/state_machine_demo.py
```

也可以指定其他配置文件，命令行参数会覆盖 YAML 中的同名参数：

```bash
python3 examples/state_machine_demo.py --config config/state_machine.yaml --kp 60 --kd 3
```

关节力矩显示也可以用命令行临时控制：

```bash
python3 examples/state_machine_demo.py --show-torque-overlay
python3 examples/state_machine_demo.py --no-show-torque-overlay
```

检查 YAML 是否生效，可以打印最终运行配置：

```bash
python3 examples/state_machine_demo.py --print-config --no-viewer
python3 examples/state_machine_demo.py --print-config --no-viewer --kp 60
```

### RS02 底层接口

RS02 真机底层控制不混入 `control/` 或 `sim/`，统一放在 `robot_io/rs02/`。默认配置文件是 `config/rs02_hardware.yaml`，其中所有真实电机默认 `enabled: false`，需要按真机 ID、方向、零位和限位确认后再逐个打开。

底层 backend dry-run 检查：

```bash
python3 examples/rs02_backend_demo.py --duration 2
python3 examples/rs02_backend_demo.py --print-config
```

状态机切换到 RS02 backend 的 dry-run 检查：

```bash
python3 examples/rs02_state_machine_dry_run.py --duration 3 --auto-stand
```

真实 RS02 输出只在底层 backend demo 中开放，并且必须显式确认：

```bash
python3 examples/rs02_backend_demo.py --live --i-understand-live-rs02
```

在未确认 12 个电机的 ID、`sign`、`zero_offset`、软件限位和急停手段前，不要打开 `enabled: true`，也不要把状态机直接跑到 live 真机。

#### RS02 真机安全边界

当前 RS02 接入只开放到底层 backend live smoke test，不开放状态机 live 真机入口。状态机切换 RS02 backend 的脚本 `rs02_state_machine_dry_run.py` 只允许 dry-run，用来验证算法层能切换底层接口，不会向真实电机发送命令。

真机相关默认安全边界如下：

- `config/rs02_hardware.yaml` 中 12 个关节默认都是 `enabled: false`。
- `rs02_backend_demo.py` 默认 dry-run，只有同时传入 `--live --i-understand-live-rs02` 才允许连接真实 RS02 总线。
- RS02 live backend 有 `command_timeout`，上层长时间不更新命令时会切到低 `kd` 阻尼命令。
- 退出、异常或 `Ctrl+C` 时会调用 `stop()`，发送零 MIT 命令并 disable 已使能电机。
- `kp/kd/torque/angle/speed` 会先经过 RS02 协议范围限幅，目标角度还会经过每个关节配置的软限位。
- 未收到新鲜反馈时，backend 不会把旧反馈当成有效闭环状态更新。

#### 上实机前必须完成的工作

这套算法上真实机器狗前，建议按下面顺序推进，不能跳过前面的硬件确认直接跑状态机或 RL：

1. 硬件断电检查：确认急停、外部断电手段、电机独立供电、CANH/CANL/GND、终端电阻、线束固定和机械限位。
2. 单电机通信检查：只使能一个电机，确认 `motor_id`、反馈角度、速度、力矩、温度和错误码能稳定读取。
3. 方向和零位标定：逐个关节确认 `sign` 和 `zero_offset`，保证 `joint_pos` 增减方向和仿真 `JOINT_ORDER` 一致。
4. 软件限位确认：为每个关节填写保守的 `joint_min/joint_max` 或 `motor_min/motor_max`，先用小范围动作验证限位不会越界。
5. 阻尼模式测试：`kp=0`，只给很小 `kd`，逐个电机确认停止、异常退出和 `Ctrl+C` 后能安全 disable。
6. 单电机小角度测试：低 `kp/kd`、低幅度目标角，从一个电机开始验证角度控制，不加载整机。
7. 单腿三电机测试：只打开一条腿，验证 hip/thigh/calf 的映射、限位、反馈和急停。
8. 12 电机只读反馈：全部电机接上但不主动输出大刚度，确认 12 路反馈稳定、无 ID 冲突、无 stale feedback。
9. 12 电机低阻尼测试：全部电机 `kp=0`、小 `kd`，验证 stop/watchdog/disable。
10. 低增益站立前检查：把仿真的站立目标转换到真实电机角度后，人工确认每个目标都在机械安全范围内。
11. 低增益静态姿态测试：只做慢速 ramp 到静态目标，不运行步态、不运行位置控制、不运行 RL。
12. 位置控制实机测试：先低速度、低步幅、低 `kp/kd`，记录 `joint_tau`、反馈延迟、饱和比例和温度。
13. RL 上实机前：必须先通过同一 backend 的 dry-run、仿真回放、动作限幅、动作斜率限制、观测 sanity check 和 emergency stop 测试。

实机测试期间建议始终保留日志：时间戳、12 关节目标角、反馈角、速度、反馈力矩、`kp/kd/tau_ff`、错误码、温度、命令延迟和反馈更新时间。

核心接口：

```python
sim.set_mit_command(kp, q_des, kd, dq_des, tau_ff)
state = sim.step()
```

控制律为：

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```
