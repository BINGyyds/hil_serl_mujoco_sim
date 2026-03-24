# hil_serl_mujoco_sim

这个目录是未来独立 GitHub 仓库 `hil_serl_mujoco_sim` 的本地形态，负责本地 MuJoCo 仿真侧能力。

说明：

- GitHub 仓库名：`hil_serl_mujoco_sim`
- 父仓挂载目录名：`sim/`
- 当前 Python 包名：`robot_controllers_mujoco`

## 包含内容

- `robot_controllers_mujoco/envs`
  - Franka MuJoCo Gymnasium 环境
- `robot_controllers_mujoco/models/franka_panda`
  - Panda MJCF 和 mesh 资产
- `robot_controllers_mujoco/spacemouse`
  - SpaceMouse 输入支持
- `robot_controllers_mujoco/robot`
  - 本地控制与离线仿真工具
- `scripts/record_demos_mujoco.py`
  - MuJoCo demo 录制

## 当前主线

默认主线模型是：

- `panda_nohand_position.xml`

同时保留并继续维护：

- `panda_nohand_effort.xml`

`panda_nohand_effort.xml` 是后续柔顺控制 / 接触控制要继续使用的正式模型，不是归档文件。

## 适用机器

- 本地 `MacBook M5`
- 主要职责：MuJoCo 仿真、viewer、demo 录制、SpaceMouse 输入

## 安装

```bash
bash setup_env_mac.sh
conda activate robot_controllers_mujoco
```

如果需要使用 `robot_controllers_mujoco.robot.*` 中依赖 Pinocchio 的控制器模块，请额外手动安装 Pinocchio。

## 基本验证

```bash
python -c "from robot_controllers_mujoco import make_reach_config, FrankaReachEnv; env = FrankaReachEnv(config=make_reach_config()); env.reset(); env.close()"
```

## 录制 demo

```bash
python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl
```

## 和训练仓的关系

这个仓库只负责仿真环境和模型，不负责 learner 训练逻辑。

如果要跑本地 actor + 远程 learner，还需要同时安装旁边的 `hil_serl_mujoco_train` 仓库。

## 推送到 GitHub

```bash
cd sim
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_sim"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_sim.git
git push -u origin main
```
