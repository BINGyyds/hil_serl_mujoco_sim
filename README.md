# robot_controllers_mujoco

`sim/` 是本工作区的 MuJoCo 仿真与示教侧仓库。它提供：

- Franka Panda MJCF 模型与资产
- `robot_controllers_mujoco` Python 包
- MuJoCo Gymnasium 环境
- SpaceMouse 输入接入
- 本地控制器与示教调试
- insertion demo 录制

## 当前范围

当前主线已经收敛到 insertion 任务，不再维护 reach 作为默认任务。仿真侧的默认基线为：

- 环境：`FrankaInsertionEnv`
- 控制后端：velocity control
- 基准模型：`panda_nohand_velocity.xml`

固定约束：

- 现在以及后续所有涉及 SpaceMouse 控制的 MuJoCo 场景，都以 `panda_nohand_velocity.xml` 为基准。

## 目录结构

- `robot_controllers_mujoco/envs`
  - MuJoCo Gymnasium 环境
  - 当前主用环境：`FrankaInsertionEnv`
- `robot_controllers_mujoco/configs`
  - 任务配置工厂
  - 当前主用配置：`make_insertion_config()`
- `robot_controllers_mujoco/models/franka_panda`
  - Panda MJCF 模型与 mesh 资源
- `robot_controllers_mujoco/common`
  - 通用配置、异步写入、数据管理
- `robot_controllers_mujoco/control`
  - task-space velocity controller、统一 teleop runtime、本地控制器封装
- `robot_controllers_mujoco/simulation`
  - MuJoCo 接口、Pinocchio 模型、轨迹/运动工具
- `robot_controllers_mujoco/spacemouse`
  - SpaceMouse 驱动与兼容层
- `scripts/record_demos_mujoco.py`
  - 使用 SpaceMouse 录制 insertion demos
- `scripts/debug/teleop_main.py`
  - 手动 teleop 调试入口

## 环境安装

### macOS

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_mac.sh
conda activate sim_env
```

### Ubuntu

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_ubuntu.sh
conda activate sim_env
```

## 基本验证

### 验证 insertion 环境

```bash
python -c "from robot_controllers_mujoco.configs.insertion_config import make_insertion_config; from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv; env = FrankaInsertionEnv(config=make_insertion_config()); obs, _ = env.reset(); print(obs['state'].keys()); env.close()"
```

### 启动本地控制器

无 SpaceMouse 调试：

```bash
python scripts/debug/teleop_main.py \
  --input-mode spacemouse \
  --control-mode velocity \
  --allow-missing-spacemouse
```

离线规划测试：

```bash
python scripts/debug/teleop_main.py \
  --input-mode planned \
  --control-mode effort
```

## 录制 insertion demos

使用真实 SpaceMouse 录制 demo：

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25
```

如果需要观察录制过程：

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25 \
  --render-human \
  --realtime
```

如果怀疑 SpaceMouse 没有生效，可以先打开输入调试：

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25 \
  --render-human \
  --realtime \
  --debug-spacemouse
```

终端会周期性打印 6D action 和按钮状态。录制时默认单条 demo 最长 1000 步，约 40 秒；需要更长时间可以加 `--max_episode_steps 1500`。如果动作太慢，可以小幅提高 `--action_scale`，例如 `--action_scale 1.5`。

录制脚本当前仅保留真实控制路径：

- 输入：SpaceMouse 6D action
- 环境：`FrankaInsertionEnv`
- 控制链：与本地 debug teleop 共享的 `TeleopRuntime` + task-space velocity controller
- action 语义：平移为世界坐标系，旋转为末端本体坐标系
- velocity 控制：显式启用 MuJoCo bias/gravity compensation
- 输出：可直接供 learner 加载的 demo `pkl`

## SpaceMouse 说明

### macOS

要求：

- 通过 Homebrew 安装 `hidapi`
- Python 环境安装 `easyhid`

排查命令：

```bash
brew --prefix hidapi
python -c "from easyhid import Enumeration; print('easyhid backend ok')"
pgrep -lf '3Dconnexion|3Dx|3dconnexion'
```

如果安装了 3Dconnexion 官方驱动，运行前建议退出相关后台进程。

### Ubuntu

要求：

- 系统安装 `hidapi`
- 普通用户访问设备时，安装 `99-spacemouse.rules`

安装 udev 规则：

```bash
sudo cp 99-spacemouse.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw
```

如果设备已经插着但仍提示 `Failed to open device`，请拔插一次 SpaceMouse 后重试。

## 与训练仓的关系

`sim/` 只负责：

- MuJoCo 仿真模型
- Gym 环境
- SpaceMouse 输入
- 本地控制器与调试
- demo 录制

分布式训练由旁边的 `train/` 负责。完整部署说明见父仓 [`README.md`](/home/lrb/Workspace/hil_serl_mujoco/README.md)。

## 常用入口

- 本地控制器：`python scripts/debug/teleop_main.py --input-mode spacemouse --control-mode velocity`
- 录制 demos：`python scripts/record_demos_mujoco.py --output demo_data/insertion_demos.pkl --num_demos 25`
