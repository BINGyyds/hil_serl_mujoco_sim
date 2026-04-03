# robot_controllers_mujoco

本目录是本地 MuJoCo 仿真侧仓库，负责：

- Franka Panda MJCF 模型
- MuJoCo Gymnasium 环境
- SpaceMouse 接入
- 本地控制器与示教调试
- 插入任务 demo 录制

当前 Python 包名是 `robot_controllers_mujoco`。

## 当前主线

当前不再以 `reach` 任务作为主线，默认任务改为：

- `FrankaInsertionEnv`
- `panda_nohand_velocity.xml`
- `SpaceMouse + velocity control`

需要固定的一条约束是：

- 现在以及后续所有涉及 SpaceMouse 控制的 MuJoCo 场景，都以 `panda_nohand_velocity.xml` 为基准

## 目录说明

- `robot_controllers_mujoco/envs`
  - MuJoCo Gymnasium 环境
  - 当前主用 `FrankaInsertionEnv`
- `robot_controllers_mujoco/configs`
  - 任务默认配置
  - 当前主用 `make_insertion_config()`
- `robot_controllers_mujoco/models/franka_panda`
  - Panda MJCF 模型与 mesh
  - 包含 `position` / `effort` / `velocity` 三套 nohand 控制模型
- `robot_controllers_mujoco/spacemouse`
  - SpaceMouse 输入驱动与兼容层
- `robot_controllers_mujoco/robot`
  - 本地控制器、Pinocchio 运动学/动力学、数据记录与视频写入
- `scripts/record_demos_mujoco.py`
  - 使用 SpaceMouse 录制 insertion demos

## 环境安装

### macOS

```bash
cd sim
bash setup_env_mac.sh
conda activate sim_env
```

### Ubuntu

```bash
cd sim
bash setup_env_ubuntu.sh
conda activate sim_env
```

## 快速验证

### 1. 验证 insertion env

```bash
python -c "from robot_controllers_mujoco.configs.insertion_config import make_insertion_config; from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv; env = FrankaInsertionEnv(config=make_insertion_config()); obs, _ = env.reset(); print(obs['state'].keys()); env.close()"
```

### 2. 启动本地控制器

无 SpaceMouse 调试：

```bash
python -m robot_controllers_mujoco.robot.main --input-mode spacemouse --control-mode velocity --allow-missing-spacemouse
```

离线规划模式：

```bash
python -m robot_controllers_mujoco.robot.main --input-mode planned --control-mode effort
```

### 3. 录制 insertion demos

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25
```

录制现在只保留真实 SpaceMouse 路径：

- 使用 `panda_nohand_velocity.xml` 作为基准场景
- `expert action -> env.step(action)` 与本地 [`robot_controller.py`](/home/lrb/Workspace/hil_serl_mujoco/sim/robot_controllers_mujoco/robot/robot_controller.py) 共用同一套 task-space velocity controller
- 正式训练前建议先录 20 到 50 条成功 demo

如果你想看录制过程：

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25 \
  --render-human \
  --realtime
```

## SpaceMouse 使用说明

### macOS

- 需要先安装 Homebrew `hidapi`
- Apple Silicon 推荐使用 patched `easyhid` fork
- 如果安装了 3Dconnexion 官方驱动，运行前建议退出 `3DconnexionHelper`
- 当前仓库内置了 macOS 下的 `hidapi` 预加载逻辑，优先尝试 `/opt/homebrew` 和 `/usr/local`

排查命令：

```bash
brew --prefix hidapi
python -c "from easyhid import Enumeration; print('easyhid backend ok')"
pgrep -lf '3Dconnexion|3Dx|3dconnexion'
```

### Ubuntu

- 需要系统安装 `hidapi`
- 如果普通用户无法读取设备，需要安装仓库内的 `99-spacemouse.rules`

安装 udev 规则：

```bash
sudo cp 99-spacemouse.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 与训练仓的关系

这个仓库只负责：

- MuJoCo 仿真模型
- Gym 环境
- SpaceMouse 输入
- 本地 demo / teleop / 调试

如果要跑本地 actor + 远程 learner，仍然需要旁边的 `train/`。

## 常用入口

- 控制器：`python -m robot_controllers_mujoco.robot.main`
- 录制 demo：`python scripts/record_demos_mujoco.py --output demo_data/insertion_demos.pkl --num_demos 25`
- 训练部署说明：看父仓 [`README.md`](/home/lrb/Workspace/hil_serl_mujoco/README.md)
