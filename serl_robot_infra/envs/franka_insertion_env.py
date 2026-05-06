"""
Franka Panda peg-in-hole insertion task in MuJoCo.

The task reuses the tool and socket geometry already shipped in the no-hand
Panda model.  Episodes start from a pre-insert pose above the socket with a
small random XY/yaw perturbation, then the policy controls the tool-tip pose
via 6-D Cartesian delta actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from serl_robot_infra.control.task_space_velocity_controller import (
    TaskSpaceVelocityController,
)
from serl_robot_infra.control.teleop_runtime import TeleopRuntime
from serl_robot_infra.envs.franka_mujoco_env import (
    FrankaMujocoConfig,
    FrankaMujocoEnv,
)
from serl_robot_infra.simulation.site_ik_solver import (
    SiteIKOptions,
    solve_site_pose_ik,
)


@dataclass
class InsertionConfig(FrankaMujocoConfig):
    """Configuration for the MuJoCo peg-in-hole insertion task."""

    socket_site_name: str = "tool_socket_center_site"

    # Reset randomisation around the pre-insert pose.
    reset_pos_noise_low: np.ndarray = field(
        default_factory=lambda: np.array([-0.01, -0.0015, 0.0])
    )
    reset_pos_noise_high: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.0015, 0.0])
    )
    reset_yaw_range: float = 0.08

    # reset 随机化阶段使用的有界非线性最小二乘 IK。
    reset_ik_max_iters: int = 80
    reset_ik_tolerance: float = 1e-4
    reset_ik_position_tolerance: float = 1e-4
    reset_ik_rotation_tolerance: float = 1e-3
    # 旧 DLS 参数仅为兼容历史 override 保留；非线性最小二乘后端不使用。
    reset_ik_step_scale: float = 0.6
    reset_ik_damping: float = 1e-4
    reset_ik_position_weight: float = 1.0
    reset_ik_rotation_weight: float = 0.05
    reset_ik_seed_weight: float = 0.01
    reset_ik_center_weight: float = 0.002

    # Reward weights.
    reward_xy_weight: float = 8.0
    reward_z_weight: float = 1.5
    reward_rot_weight: float = 0.3
    reward_depth_weight: float = 2.0
    reward_success_bonus: float = 2000.0
    reward_failure_penalty: float = 25.0
    reward_step_penalty: float = 0.01
    reward_depth_xy_threshold: float = 0.004
    reward_depth_rot_threshold: float = 0.12
    # Reserved for potential incremental depth shaping; currently unused.
    reward_depth_deadband: float = 0.0002

    # Success thresholds.
    success_xy_threshold: float = 0.002
    success_rot_threshold: float = 0.1
    success_z_gap_threshold: float = 0.002
    success_linear_velocity_threshold: float = 0.01
    success_angular_velocity_threshold: float = 0.05
    success_tip_z_threshold: float = 0.105

    # Failure / workspace bounds.
    workspace_low: np.ndarray = field(
        default_factory=lambda: np.array([0.49, -0.05, -0.02])
    )
    workspace_high: np.ndarray = field(
        default_factory=lambda: np.array([0.62, 0.05, 0.12])
    )
    failure_xy_threshold: float = 0.012
    failure_z_gap_threshold: float = 0.012
    failure_tip_z_threshold: float = 0.075

    max_episode_steps: int = 1000

    # Shared task-space velocity controller parameters.
    input_deadzone: float = 0.05
    input_filter_alpha: float = 0.25
    max_reference_linear_velocity: float = 0.05
    max_reference_angular_velocity: float = 0.10
    pose_error_gain: float = 25.0
    ik_damping: float = 1e-4
    nullspace_gain: float = 0.15
    joint_velocity_limits: Tuple[float, ...] = (
        2.175,
        2.175,
        2.175,
        2.175,
        2.610,
        2.610,
        2.610,
    )
    joint_limit_margin: float = 0.05
    enable_admittance: bool = False
    wrench_filter_alpha: float = 0.2
    force_deadband: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    torque_deadband: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    max_force_world: Tuple[float, float, float] = (20.0, 20.0, 20.0)
    max_torque_world: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    admittance_mass: Tuple[float, ...] = (
        2.0,
        2.0,
        3.0,
        0.15,
        0.15,
        0.15,
    )
    admittance_damping: Tuple[float, ...] = (
        80.0,
        80.0,
        110.0,
        8.0,
        8.0,
        8.0,
    )
    admittance_stiffness: Tuple[float, ...] = (
        80.0,
        80.0,
        140.0,
        10.0,
        10.0,
        10.0,
    )
    max_compliance_linear_offset: float = 0.02
    max_compliance_angular_offset: float = 0.2
    max_compliance_linear_velocity: float = 0.10
    max_compliance_angular_velocity: float = 0.5
    joint_velocity_filter_alpha: float = 1.0


class FrankaInsertionEnv(FrankaMujocoEnv):
    """Simple pre-aligned peg-in-hole insertion task."""

    def __init__(
        self,
        config: Optional[InsertionConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.insertion_cfg = config or InsertionConfig()
        super().__init__(config=self.insertion_cfg, render_mode=render_mode)

        self._socket_site_id = self.model.site(self.insertion_cfg.socket_site_name).id
        self._step_count = 0
        self._ik_data = mujoco.MjData(self.model)
        self._last_reset_ik_q: Optional[np.ndarray] = None
        self._task_space_controller = TaskSpaceVelocityController(
            input_deadzone=self.insertion_cfg.input_deadzone,
            input_filter_alpha=self.insertion_cfg.input_filter_alpha,
            max_reference_linear_velocity=self.insertion_cfg.max_reference_linear_velocity,
            max_reference_angular_velocity=self.insertion_cfg.max_reference_angular_velocity,
            pose_error_gain=self.insertion_cfg.pose_error_gain,
            ik_damping=self.insertion_cfg.ik_damping,
            nullspace_gain=self.insertion_cfg.nullspace_gain,
            joint_velocity_limits=np.asarray(
                self.insertion_cfg.joint_velocity_limits,
                dtype=np.float64,
            ),
            joint_limit_margin=self.insertion_cfg.joint_limit_margin,
            lower_position_limits=self._q_low,
            upper_position_limits=self._q_high,
            enable_admittance=self.insertion_cfg.enable_admittance,
            wrench_filter_alpha=self.insertion_cfg.wrench_filter_alpha,
            force_deadband=np.asarray(self.insertion_cfg.force_deadband, dtype=np.float64),
            torque_deadband=np.asarray(self.insertion_cfg.torque_deadband, dtype=np.float64),
            max_force_world=np.asarray(self.insertion_cfg.max_force_world, dtype=np.float64),
            max_torque_world=np.asarray(self.insertion_cfg.max_torque_world, dtype=np.float64),
            admittance_mass=np.asarray(self.insertion_cfg.admittance_mass, dtype=np.float64),
            admittance_damping=np.asarray(self.insertion_cfg.admittance_damping, dtype=np.float64),
            admittance_stiffness=np.asarray(self.insertion_cfg.admittance_stiffness, dtype=np.float64),
            max_compliance_linear_offset=self.insertion_cfg.max_compliance_linear_offset,
            max_compliance_angular_offset=self.insertion_cfg.max_compliance_angular_offset,
            max_compliance_linear_velocity=self.insertion_cfg.max_compliance_linear_velocity,
            max_compliance_angular_velocity=self.insertion_cfg.max_compliance_angular_velocity,
            joint_velocity_filter_alpha=self.insertion_cfg.joint_velocity_filter_alpha,
        )
        self._teleop_runtime = TeleopRuntime(
            controller=self._task_space_controller,
            get_control_state=self._get_teleop_control_state,
            set_joint_velocity=self._set_joint_velocity_command,
            step_simulation=self._step_substeps,
            control_dt=self.cfg.control_dt,
            enable_admittance=self.insertion_cfg.enable_admittance,
        )

        state_spaces = dict(self._base_state_spaces)
        state_spaces["socket_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        state_spaces["relative_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        state_spaces["insertion_depth"] = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float64
        )
        state_spaces["controller_target_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(
            {"state": spaces.Dict(state_spaces)}
        )

        self._reset_to_home_state()

        self._preinsert_tip_pos, self._target_tip_rot = self._get_ee_pose()
        self._socket_pos, self._socket_rot = self._get_socket_pose()
        self._max_insertion_depth = max(
            float(self._preinsert_tip_pos[2] - self._socket_pos[2]), 1e-6
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        """重置到带小扰动的预插入初始位姿。

        父类 reset 先恢复 MJCF 的 home keyframe。本子类随后采样一个任务空间
        小扰动，用有界 IK 将 ``tool_tip_site`` 放到目标位姿，最后把速度控制器
        的内部目标同步到新的真实仿真状态。
        """
        obs, info = super().reset(seed=seed, options=options)
        self._step_count = 0
        self._socket_pos, self._socket_rot = self._get_socket_pose()

        target_tip_pos, target_tip_rot = self._sample_reset_tip_pose()
        self._set_tip_pose_with_ik(target_tip_pos, target_tip_rot)
        self._sync_controller_reference_to_current_state()

        obs = self._get_obs()
        info = self._get_info(obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self._teleop_runtime.step(
            action,
            dt=self.cfg.control_dt,
        )

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = False
        info = self._get_info(obs)

        self._step_count += 1
        if self._step_count >= self.insertion_cfg.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        obs = super()._get_obs()
        tip_pos, tip_rot = self._get_ee_pose()
        socket_pos, socket_rot = self._get_socket_pose()

        socket_quat = Rotation.from_matrix(socket_rot).as_quat()
        relative_quat = Rotation.from_matrix(socket_rot @ tip_rot.T).as_quat()
        relative_pos = socket_pos - tip_pos

        insertion_depth = np.array(
            [
                np.clip(
                    self._preinsert_tip_pos[2] - tip_pos[2],
                    0.0,
                    self._max_insertion_depth,
                )
            ],
            dtype=np.float64,
        )

        obs["state"]["socket_pose"] = np.concatenate([socket_pos, socket_quat])
        obs["state"]["relative_pose"] = np.concatenate([relative_pos, relative_quat])
        obs["state"]["insertion_depth"] = insertion_depth
        obs["state"]["controller_target_pose"] = self._get_controller_target_pose()
        return obs

    def _compute_reward(self, obs: Dict) -> float:
        tip_pos, tip_rot = self._get_ee_pose()
        socket_pos, _ = self._get_socket_pose()
        cfg = self.insertion_cfg

        xy_error = float(np.linalg.norm(tip_pos[:2] - socket_pos[:2]))
        z_gap = float(max(tip_pos[2] - socket_pos[2], 0.0))
        rot_error = self._compute_rotation_error(tip_rot)
        current_insertion_depth = float(obs["state"]["insertion_depth"][0])
        normalized_depth = current_insertion_depth / self._max_insertion_depth

        aligned_for_depth = (
            xy_error < cfg.reward_depth_xy_threshold
            and rot_error < cfg.reward_depth_rot_threshold
        )
        if aligned_for_depth:
            depth_reward = cfg.reward_depth_weight * normalized_depth
        else:
            depth_reward = 0.0

        reward = (
            -cfg.reward_xy_weight * xy_error
            - cfg.reward_z_weight * z_gap
            - cfg.reward_rot_weight * rot_error
            + depth_reward
            - cfg.reward_step_penalty
        )

        if self._is_success(tip_pos, tip_rot):
            reward += cfg.reward_success_bonus
        elif self._is_failure(tip_pos):
            reward -= cfg.reward_failure_penalty

        return float(reward)

    def _check_termination(self, obs: Dict) -> bool:
        tip_pos, tip_rot = self._get_ee_pose()
        return bool(self._is_success(tip_pos, tip_rot) or self._is_failure(tip_pos))

    def _get_info(self, obs: Dict) -> Dict[str, Any]:
        tip_pos, tip_rot = self._get_ee_pose()
        tip_lin_vel, tip_ang_vel = self._get_ee_velocity()
        socket_pos, _ = self._get_socket_pose()

        xy_error = float(np.linalg.norm(tip_pos[:2] - socket_pos[:2]))
        z_gap = float(max(tip_pos[2] - socket_pos[2], 0.0))
        rot_error = self._compute_rotation_error(tip_rot)
        linear_speed = float(np.linalg.norm(tip_lin_vel))
        angular_speed = float(np.linalg.norm(tip_ang_vel))
        insertion_depth = float(
            np.clip(
                self._preinsert_tip_pos[2] - tip_pos[2],
                0.0,
                self._max_insertion_depth,
            )
        )
        success = self._is_success(tip_pos, tip_rot)
        failure = self._is_failure(tip_pos)

        return {
            "xy_error": xy_error,
            "z_gap": z_gap,
            "rot_error": rot_error,
            "tcp_linear_speed": linear_speed,
            "tcp_angular_speed": angular_speed,
            "insertion_depth": insertion_depth,
            "success": success,
            "failure": failure,
        }

    def _sample_reset_tip_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """围绕名义 home 末端位姿采样预插入 reset 位姿。"""
        cfg = self.insertion_cfg
        pos_noise = self.np_random.uniform(
            cfg.reset_pos_noise_low,
            cfg.reset_pos_noise_high,
        )
        yaw_noise = float(
            self.np_random.uniform(-cfg.reset_yaw_range, cfg.reset_yaw_range)
        )

        target_tip_pos = self._preinsert_tip_pos + pos_noise
        target_tip_pos = np.clip(
            target_tip_pos,
            cfg.workspace_low,
            cfg.workspace_high,
        )
        target_tip_rot = (
            Rotation.from_euler("z", yaw_noise).as_matrix() @ self._target_tip_rot
        )
        return target_tip_pos, target_tip_rot

    def _set_tip_pose_with_ik(
        self,
        target_tip_pos: np.ndarray,
        target_tip_rot: np.ndarray,
    ) -> None:
        """使用有界 site 位姿 IK 将仿真机械臂移动到 reset 位姿。

        该函数只在 episode 初始化时使用。运行时 action 仍然经过
        :class:`TaskSpaceVelocityController` 及其速度级 DLS 求解，因此修改该
        函数不会改变每一步 policy 控制逻辑。
        """
        cfg = self.insertion_cfg
        target_transform = np.eye(4, dtype=np.float64)
        target_transform[:3, :3] = target_tip_rot
        target_transform[:3, 3] = target_tip_pos

        q_current = self.data.qpos[self._qpos_addrs].copy()
        finite_limits = np.isfinite(self._q_low) & np.isfinite(self._q_high)
        q_center = q_current.copy()
        q_center[finite_limits] = 0.5 * (
            self._q_low[finite_limits] + self._q_high[finite_limits]
        )

        extra_seeds = [q_center]
        if self._last_reset_ik_q is not None:
            extra_seeds.insert(0, self._last_reset_ik_q)

        self._ik_data.qpos[:] = self.data.qpos
        self._ik_data.qvel[:] = self.data.qvel
        result = solve_site_pose_ik(
            model=self.model,
            data=self._ik_data,
            target_transform=target_transform,
            q_seed=q_current,
            qpos_addrs=self._qpos_addrs,
            dof_addrs=self._dof_addrs,
            q_low=self._q_low,
            q_high=self._q_high,
            site_id=self._ee_site_id,
            options=SiteIKOptions(
                max_iterations=cfg.reset_ik_max_iters,
                tolerance=cfg.reset_ik_tolerance,
                position_tolerance=cfg.reset_ik_position_tolerance,
                rotation_tolerance=cfg.reset_ik_rotation_tolerance,
                position_weight=cfg.reset_ik_position_weight,
                rotation_weight=cfg.reset_ik_rotation_weight,
                seed_weight=cfg.reset_ik_seed_weight,
                center_weight=cfg.reset_ik_center_weight,
            ),
            extra_seeds=extra_seeds,
        )
        self.data.qpos[self._qpos_addrs] = result.q
        self.data.qvel[self._dof_addrs] = 0.0
        self._last_reset_ik_q = result.q.copy()
        self._sync_actuator_ctrl_to_current_state()
        mujoco.mj_forward(self.model, self.data)

    def _get_socket_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self._socket_site_id].copy()
        rot = self.data.site_xmat[self._socket_site_id].reshape(3, 3).copy()
        return pos, rot

    def _get_current_transform(self) -> np.ndarray:
        pos, rot = self._get_ee_pose()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        return transform

    def _get_site_jacobian(self) -> np.ndarray:
        """返回当前 tip LOCAL 坐标系下的 ``tool_tip_site`` Jacobian。

        MuJoCo 给出的是世界系 site Jacobian。共享速度控制器计算的是局部位姿
        误差，因此在求解关节速度前需要把 Jacobian 转到同一个坐标系。
        """
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        jacobian_world = np.vstack([
            jacp[:, self._dof_addrs],
            jacr[:, self._dof_addrs],
        ])
        _, tip_rot = self._get_ee_pose()
        world_to_local = np.block([
            [tip_rot.T, np.zeros((3, 3), dtype=np.float64)],
            [np.zeros((3, 3), dtype=np.float64), tip_rot.T],
        ])
        return world_to_local @ jacobian_world

    def _get_teleop_control_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_current = self.data.qpos[self._qpos_addrs].copy()
        current_transform = self._get_current_transform()
        jacobian = self._get_site_jacobian()
        return q_current, current_transform, jacobian

    def _set_joint_velocity_command(self, joint_velocity_command: np.ndarray) -> None:
        self.data.ctrl[self._actuator_ids] = np.asarray(
            joint_velocity_command,
            dtype=np.float64,
        )

    def _sync_controller_reference_to_current_state(self) -> None:
        self._teleop_runtime.reset_reference()

    def _get_controller_target_pose(self) -> np.ndarray:
        target_transform = self._task_space_controller.get_target_transform()
        if target_transform is None:
            target_transform = self._get_current_transform()
        target_quat = Rotation.from_matrix(target_transform[:3, :3]).as_quat()
        return np.concatenate([target_transform[:3, 3], target_quat])

    def _compute_rotation_error(self, tip_rot: np.ndarray) -> float:
        rotation_delta = self._target_tip_rot.T @ tip_rot
        cos_angle = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _is_success(self, tip_pos: np.ndarray, tip_rot: np.ndarray) -> bool:
        cfg = self.insertion_cfg
        tip_lin_vel, tip_ang_vel = self._get_ee_velocity()
        xy_error = np.linalg.norm(tip_pos[:2] - self._socket_pos[:2])
        z_gap = max(float(tip_pos[2] - self._socket_pos[2]), 0.0)
        rot_error = self._compute_rotation_error(tip_rot)
        linear_speed = np.linalg.norm(tip_lin_vel)
        angular_speed = np.linalg.norm(tip_ang_vel)
        return bool(
            xy_error < cfg.success_xy_threshold
            and rot_error < cfg.success_rot_threshold
            and z_gap < cfg.success_z_gap_threshold
            and linear_speed < cfg.success_linear_velocity_threshold
            and angular_speed < cfg.success_angular_velocity_threshold
        )

    def _is_failure(self, tip_pos: np.ndarray) -> bool:
        cfg = self.insertion_cfg
        out_of_workspace = np.any(tip_pos < cfg.workspace_low) or np.any(
            tip_pos > cfg.workspace_high
        )
        xy_error = np.linalg.norm(tip_pos[:2] - self._socket_pos[:2])
        z_gap = max(float(tip_pos[2] - self._socket_pos[2]), 0.0)
        obvious_misalignment = (
            z_gap < cfg.failure_z_gap_threshold
            and xy_error > cfg.failure_xy_threshold
        )
        return bool(out_of_workspace or obvious_misalignment)
