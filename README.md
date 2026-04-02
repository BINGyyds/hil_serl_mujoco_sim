# robot_controllers_mujoco

本目录是本地 MuJoCo 仿真侧仓库，当前 Python 包名为 `robot_controllers_mujoco`。  
它负责 Franka Panda 的 MuJoCo 模型、Gymnasium 环境、SpaceMouse 接入、以及本地控制/示教工具；不负责远程 learner 或强化学习训练主逻辑。

## 目录说明

- `robot_controllers_mujoco/envs`
  - MuJoCo Gymnasium 环境
  - 当前提供 `FrankaMujocoEnv` 和 `FrankaReachEnv`
- `robot_controllers_mujoco/configs`
  - 任务默认配置
  - 当前提供 `make_reach_config()`
- `robot_controllers_mujoco/models/franka_panda`
  - Panda MJCF 模型与 mesh 资产
  - 包含 `position` / `effort` / `velocity` 三套 nohand 控制模型
- `robot_controllers_mujoco/spacemouse`
  - SpaceMouse 输入驱动与兼容层
- `robot_controllers_mujoco/robot`
  - 本地控制器、Pinocchio 运动学/动力学、数据记录与视频写入
- `scripts/record_demos_mujoco.py`
  - 使用 SpaceMouse 在 MuJoCo 中录制 demo

## 当前能力

这个仓库现在包含两条主线：

1. Gym 环境主线
   - `FrankaReachEnv`
   - 适合本地验证和对接训练仓

2. 本地控制器主线
   - `python -m robot_controllers_mujoco.robot.main`
   - 支持 `planned + position/effort`
   - 支持 `spacemouse + velocity`

目前默认使用的 Panda nohand 模型有：

- `panda_nohand_position.xml`
- `panda_nohand_effort.xml`
- `panda_nohand_velocity.xml`

## 环境安装

### macOS

```bash
cd sim
bash setup_env_mac.sh
conda activate robot_controllers_mujoco
```

### Ubuntu

```bash
cd sim
bash setup_env_ubuntu.sh
conda activate robot_controllers_mujoco
```

这两个脚本都会做这些事情：

- 创建 Conda 环境
- 安装本仓库 Python 依赖
- 安装 `pinocchio`
- 安装 SpaceMouse 依赖所需的系统库
- 以 editable 方式安装 `robot_controllers_mujoco`

## 快速验证

### 1. 验证 Gym 环境

```bash
python -c "from robot_controllers_mujoco import FrankaReachEnv, make_reach_config; env = FrankaReachEnv(config=make_reach_config()); obs, _ = env.reset(); print(obs['state'].keys()); env.close()"
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

### 3. 录制 MuJoCo demo

```bash
python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl
```

## SpaceMouse 使用说明

### macOS

- 需要先安装 Homebrew `hidapi`
- Apple Silicon 推荐使用 patched `easyhid` fork
- 如果安装了 3Dconnexion 官方驱动，运行前建议退出 `3DconnexionHelper`
- 当前仓库内置了 macOS 下的 `hidapi` 预加载逻辑，优先尝试 `/opt/homebrew` 和 `/usr/local`

可用这些命令排查：

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

如果要跑本地 actor + 远程 learner，仍然需要旁边的训练仓。

建议分工是：

- `sim/`：本地仿真、SpaceMouse、demo、环境定义
- `train/`：actor / learner、buffer、训练配置、checkpoint

## 常用入口

- 环境：`python -c "from robot_controllers_mujoco import FrankaReachEnv, make_reach_config; ..."`
- 控制器：`python -m robot_controllers_mujoco.robot.main`
- 录制 demo：`python scripts/record_demos_mujoco.py --output ...`

## 输出目录

运行过程中默认会写到：

- `output/videos`
- `output/docs`

如果你后面把这个目录单独拆成 GitHub 仓库，建议继续保留当前目录结构和脚本入口，不要再改包名。
