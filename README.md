# robot_controllers_mujoco

`sim/` 负责 MuJoCo 仿真、SpaceMouse 调试、demo 录制，以及相机渲染相关能力。

当前主线：

- 环境：`FrankaInsertionEnv`
- 模型：`panda_nohand_velocity.xml`
- 控制链：SpaceMouse + task-space velocity control

## 安装

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/sim
./setup_env_sim.sh
conda activate sim_env
```

## 调试入口

```bash
python scripts/debug/teleop_main.py --allow-missing-spacemouse
```

带末端相机窗口：

```bash
python scripts/debug/teleop_main.py --allow-missing-spacemouse --show-cameras
```

## 录制 demos

```bash
python scripts/record_demos_mujoco.py \
  --output demo_data/insertion_demos.pkl \
  --num_demos 25 \
  --render-human \
  --realtime \
  --show-cameras
```

默认录制参数与训练主线对齐：

- 控制频率 `100 Hz`
- `sim_timestep=0.001`
- `n_substeps=10`

录制日志会打印：

- `xy_error`
- `z_gap`
- `rot_error`
- `linear_speed`
- `angular_speed`

## 相机与视觉扩展

当前保留的视觉相关能力：

- velocity MJCF 中的 debug / eval cameras
- `--show-cameras`
- `MujocoInterface.render_*`
- eval 视频相机选择逻辑

这部分可以直接支撑后续把末端或场景相机画面接进 RL observation。

## 说明

分布式训练由旁边的 `train/` 负责。完整端到端流程见父仓 [README.md](/home/lrb/Workspace/hil_serl_mujoco/README.md)。
