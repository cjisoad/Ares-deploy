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
- `examples/`：示例目录。
  - `mit_control_demo.py` 演示如何直接调用 Python API 下发 MIT 控制指令并运行仿真。
  - `suspended_mit_demo.py` 演示机身固定在离地 1.5 米、四肢自然下垂并使用 MIT 力矩控制的场景。
  - `free_fall_demo.py` 演示机身从离地 1 米自然下落至蓝色棋盘格地面的场景。
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
python3 examples/free_fall_demo.py
```

核心接口：

```python
sim.set_mit_command(kp, q_des, kd, dq_des, tau_ff)
state = sim.step()
```

控制律为：

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```
