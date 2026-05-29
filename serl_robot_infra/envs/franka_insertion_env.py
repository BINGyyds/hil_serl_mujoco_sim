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

from serl_robot_infra.common.se3_utils import (
    compose_transforms,
    exp_so3,
    invert_transform,
    log_se3,
    log_so3,
    make_transform,
    rotation_matrix_to_quat_wxyz,
)
from serl_robot_infra.control.nric_torque_controller import NRICTorqueController
from serl_robot_infra.control.se3_torque_controller import SE3TorqueController
from serl_robot_infra.control.se3_torque_runtime import SE3TorqueRuntime
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
    task_frame_site_name: str = "tool_socket_center_site"
    control_frame_site_name: str = "tool_mount"
    nominal_pinocchio_mjcf_path: str = ""
    nominal_pinocchio_frame_name: str = "attachment"

    # Reset randomisation around the pre-insert pose.
    reset_pos_noise_low: np.ndarray = field(
        default_factory=lambda: np.array([-0.012, -0.003, 0.0])
    )
    reset_pos_noise_high: np.ndarray = field(
        default_factory=lambda: np.array([0.012, 0.003, 0.0])
    )
    reset_yaw_range: float = 0.12

    # Task-case randomisation used for evaluation / transfer testing.
    eval_case_xy_low: np.ndarray = field(
        default_factory=lambda: np.array([-0.012, -0.003])
    )
    eval_case_xy_high: np.ndarray = field(
        default_factory=lambda: np.array([0.012, 0.003])
    )
    eval_case_yaw_range: float = 0.12

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

    # Legacy dense reward weights retained for backward-compatible overrides.
    reward_xy_weight: float = 8.0
    reward_z_weight: float = 1.5
    reward_rot_weight: float = 1.2
    reward_body_pos_weight: float = 1.0
    reward_body_rot_weight: float = 3.0
    reward_linear_speed_weight: float = 0.15
    reward_angular_speed_weight: float = 0.10
    reward_lateral_force_weight: float = 0.20
    reward_lateral_torque_weight: float = 0.10
    reward_depth_weight: float = 2.0
    reward_success_bonus: float = 2000.0
    reward_failure_penalty: float = 25.0
    reward_step_penalty: float = 0.01
    reward_depth_xy_threshold: float = 0.004
    reward_depth_rot_threshold: float = 0.1
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
    max_reference_linear_offset: float = 0.01
    max_reference_angular_offset: float = 0.2
    wrench_force_scale: float = 20.0
    wrench_torque_scale: float = 1.0
    ik_damping: float = 1e-4
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
    use_external_wrench_feedback: bool = False
    enable_nric: bool = True
    impedance_inertia: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
    impedance_stiffness: Tuple[float, ...] = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0)
    impedance_damping: Tuple[float, ...] = (
        10.0,
        10.0,
        10.0,
        1.0,
        1.0,
        1.0,
    )
    nullspace_stiffness: float = 10.0
    nullspace_damping: float = 2.0
    nric_k_gamma: Tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    nric_k_p: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    nric_k_i: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    nric_integral_limit: float = 0.2
    nric_integral_deadband: float = 5e-3
    nric_integral_leak_rate: float = 0.2
    nric_zero_cross_reset_ratio: float = 0.2


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
        self._task_frame_site_id = self.model.site(
            self.insertion_cfg.task_frame_site_name
        ).id
        self._socket_body_id = int(self.model.site(self._task_frame_site_id).bodyid)
        self._step_count = 0
        self._ik_data = mujoco.MjData(self.model)
        self._last_reset_ik_q: Optional[np.ndarray] = None
        self._default_socket_body_pos = self.model.body_pos[self._socket_body_id].copy()
        self._default_socket_body_quat = self.model.body_quat[
            self._socket_body_id
        ].copy()
        self._default_socket_body_rot = self._body_quat_wxyz_to_rotation(
            self._default_socket_body_quat
        )
        self._current_task_case_xy = np.zeros(2, dtype=np.float64)
        self._current_task_case_yaw = 0.0
        self._current_reset_case_pos_noise = np.zeros(3, dtype=np.float64)
        self._current_reset_case_yaw = 0.0
        self._control_frame_site_id = self.model.site(
            self.insertion_cfg.control_frame_site_name
        ).id
        self._control_wrench_bias_local = np.zeros(6, dtype=np.float64)

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
        state_spaces["pose_error_body"] = spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype=np.float64
        )
        state_spaces["twist_body"] = spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype=np.float64
        )
        state_spaces["wrench_control_body"] = spaces.Box(
            -1.0, 1.0, shape=(6,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(
            {"state": spaces.Dict(state_spaces)}
        )

        self._reset_to_home_state()
        self._tool_tip_to_control_transform = compose_transforms(
            invert_transform(self._get_current_transform()),
            self._get_control_frame_transform(),
        )
        self._task_space_controller = SE3TorqueController(
            apparent_inertia=np.asarray(self.insertion_cfg.impedance_inertia, dtype=np.float64),
            lambda_stiffness=np.asarray(self.insertion_cfg.impedance_stiffness, dtype=np.float64),
            physical_damping=np.asarray(self.insertion_cfg.impedance_damping, dtype=np.float64),
            q_nominal=self.data.qpos[self._qpos_addrs].copy(),
            nullspace_stiffness=self.insertion_cfg.nullspace_stiffness,
            nullspace_damping=self.insertion_cfg.nullspace_damping,
        )
        self._nric_controller = NRICTorqueController(
            nominal_mjcf_path=self.insertion_cfg.nominal_pinocchio_mjcf_path,
            end_effector_frame_name=self.insertion_cfg.nominal_pinocchio_frame_name,
            time_step=self.cfg.control_dt,
            gravity_z=float(self.model.opt.gravity[2]),
            enabled=self.insertion_cfg.enable_nric,
            q_nominal=self.data.qpos[self._qpos_addrs].copy(),
            apparent_inertia=np.asarray(self.insertion_cfg.impedance_inertia, dtype=np.float64),
            lambda_stiffness=np.asarray(self.insertion_cfg.impedance_stiffness, dtype=np.float64),
            physical_damping=np.asarray(self.insertion_cfg.impedance_damping, dtype=np.float64),
            nullspace_stiffness=self.insertion_cfg.nullspace_stiffness,
            nullspace_damping=self.insertion_cfg.nullspace_damping,
            k_gamma=np.asarray(self.insertion_cfg.nric_k_gamma, dtype=np.float64),
            k_p=np.asarray(self.insertion_cfg.nric_k_p, dtype=np.float64),
            k_i=np.asarray(self.insertion_cfg.nric_k_i, dtype=np.float64),
            integral_limit=self.insertion_cfg.nric_integral_limit,
            integral_deadband=self.insertion_cfg.nric_integral_deadband,
            integral_leak_rate=self.insertion_cfg.nric_integral_leak_rate,
            zero_cross_reset_ratio=self.insertion_cfg.nric_zero_cross_reset_ratio,
        )
        self._teleop_runtime = SE3TorqueRuntime(
            controller=self._task_space_controller,
            nric_controller=self._nric_controller,
            get_control_state=self._get_teleop_control_state,
            set_joint_torque=self._set_joint_torque_command,
            step_simulation=self._step_substeps,
            control_dt=self.cfg.control_dt,
            sim_steps_per_control=1,
            tip_to_control_transform=self._tool_tip_to_control_transform,
            input_deadzone=self.insertion_cfg.input_deadzone,
            input_filter_alpha=self.insertion_cfg.input_filter_alpha,
            max_reference_linear_velocity=self.insertion_cfg.max_reference_linear_velocity,
            max_reference_angular_velocity=self.insertion_cfg.max_reference_angular_velocity,
            max_reference_linear_offset=self.insertion_cfg.max_reference_linear_offset,
            max_reference_angular_offset=self.insertion_cfg.max_reference_angular_offset,
            use_external_wrench_feedback=self.insertion_cfg.use_external_wrench_feedback,
        )

        self._preinsert_tip_pos, self._default_goal_tip_rot = self._get_ee_pose()
        self._socket_pos, self._socket_rot = self._get_socket_pose()
        _, default_task_rot = self._get_task_frame_pose()
        self._default_task_to_goal_rot = default_task_rot.T @ self._default_goal_tip_rot
        self._default_goal_tip_transform = make_transform(
            rotation=self._default_goal_tip_rot,
            translation=self._socket_pos,
        )
        self._default_preinsert_tip_transform = make_transform(
            rotation=self._default_goal_tip_rot,
            translation=self._preinsert_tip_pos,
        )
        self._default_preinsert_from_goal = compose_transforms(
            invert_transform(self._default_goal_tip_transform),
            self._default_preinsert_tip_transform,
        )
        self._goal_tip_transform = self._default_goal_tip_transform.copy()
        self._preinsert_tip_transform = self._default_preinsert_tip_transform.copy()
        self._goal_frame_preinsert_translation = np.zeros(3, dtype=np.float64)
        self._axial_reference_sign = 1.0
        self._max_insertion_depth = 1e-6
        self._update_task_geometry_state()

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
        self._configure_task_case(options)

        target_tip_pos, target_tip_rot = self._sample_reset_tip_pose()
        self._set_tip_pose_with_ik(target_tip_pos, target_tip_rot)
        self._tare_control_wrench()
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

        pose_error_body = self._compute_pose_error_body(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )
        twist_body = self._get_ee_body_twist()
        wrench_control_body = self._get_normalized_control_frame_wrench()
        _, _, _, insertion_depth = self._compute_goal_frame_errors(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )

        obs["state"]["socket_pose"] = np.concatenate([socket_pos, socket_quat])
        obs["state"]["relative_pose"] = np.concatenate([relative_pos, relative_quat])
        obs["state"]["insertion_depth"] = np.array([insertion_depth], dtype=np.float64)
        obs["state"]["controller_target_pose"] = self._get_controller_target_pose()
        obs["state"]["pose_error_body"] = pose_error_body
        obs["state"]["twist_body"] = twist_body
        obs["state"]["wrench_control_body"] = wrench_control_body
        return obs

    def _compute_reward(self, obs: Dict) -> float:
        tip_pos, tip_rot = self._get_ee_pose()
        cfg = self.insertion_cfg
        pose_error_body = np.asarray(obs["state"]["pose_error_body"], dtype=np.float64)
        twist_body = np.asarray(obs["state"]["twist_body"], dtype=np.float64)
        wrench_control_body = np.asarray(
            obs["state"]["wrench_control_body"],
            dtype=np.float64,
        )

        e_p = float(np.linalg.norm(pose_error_body[:3]))
        e_r = float(np.linalg.norm(pose_error_body[3:]))
        v_lin = float(np.linalg.norm(twist_body[:3]))
        v_ang = float(np.linalg.norm(twist_body[3:]))
        f_lat = float(np.linalg.norm(wrench_control_body[:2]))
        tau_lat = float(np.linalg.norm(wrench_control_body[3:5]))

        xy_error, z_gap, rot_error, _ = self._compute_goal_frame_errors(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )
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
            - cfg.reward_body_pos_weight * e_p
            - cfg.reward_body_rot_weight * e_r
            - cfg.reward_linear_speed_weight * v_lin
            - cfg.reward_angular_speed_weight * v_ang
            - cfg.reward_lateral_force_weight * f_lat
            - cfg.reward_lateral_torque_weight * tau_lat
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
        pose_error_body = np.asarray(obs["state"]["pose_error_body"], dtype=np.float64)
        wrench_control_body = np.asarray(
            obs["state"]["wrench_control_body"],
            dtype=np.float64,
        )
        xy_error, z_gap, rot_error, insertion_depth = self._compute_goal_frame_errors(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )
        linear_speed = float(np.linalg.norm(tip_lin_vel))
        angular_speed = float(np.linalg.norm(tip_ang_vel))
        body_pos_error = float(np.linalg.norm(pose_error_body[:3]))
        body_rot_error = float(np.linalg.norm(pose_error_body[3:]))
        wrench_lateral_force = float(np.linalg.norm(wrench_control_body[:2]))
        wrench_lateral_torque = float(np.linalg.norm(wrench_control_body[3:5]))
        success = self._is_success(tip_pos, tip_rot)
        failure = self._is_failure(tip_pos)

        return {
            "xy_error": xy_error,
            "z_gap": z_gap,
            "rot_error": rot_error,
            "body_pos_error": body_pos_error,
            "body_rot_error": body_rot_error,
            "wrench_lateral_force": wrench_lateral_force,
            "wrench_lateral_torque": wrench_lateral_torque,
            "tcp_linear_speed": linear_speed,
            "tcp_angular_speed": angular_speed,
            "insertion_depth": insertion_depth,
            "success": success,
            "failure": failure,
            "task_case_x": float(self._current_task_case_xy[0]),
            "task_case_y": float(self._current_task_case_xy[1]),
            "task_case_yaw": float(self._current_task_case_yaw),
            "reset_case_x": float(self._current_reset_case_pos_noise[0]),
            "reset_case_y": float(self._current_reset_case_pos_noise[1]),
            "reset_case_z": float(self._current_reset_case_pos_noise[2]),
            "reset_case_yaw": float(self._current_reset_case_yaw),
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
        self._current_reset_case_pos_noise = np.asarray(
            pos_noise,
            dtype=np.float64,
        ).reshape((3,))
        self._current_reset_case_yaw = yaw_noise

        target_tip_pos = self._preinsert_tip_transform[:3, 3] + pos_noise
        target_tip_pos = np.clip(
            target_tip_pos,
            cfg.workspace_low,
            cfg.workspace_high,
        )
        target_tip_rot = (
            self._preinsert_tip_transform[:3, :3]
            @ Rotation.from_euler("z", yaw_noise).as_matrix()
        )
        return target_tip_pos, target_tip_rot

    def _set_tip_pose_with_ik(
        self,
        target_tip_pos: np.ndarray,
        target_tip_rot: np.ndarray,
    ) -> None:
        """使用有界 site 位姿 IK 将仿真机械臂移动到 reset 位姿。

        该函数只在 episode 初始化时使用。运行时 action 会进入当前的
        SE(3) torque controller + NRIC 主链，因此修改该函数不会改变每一步
        policy 控制逻辑。
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

    def _get_task_frame_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self._task_frame_site_id].copy()
        rot = self.data.site_xmat[self._task_frame_site_id].reshape(3, 3).copy()
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

    def _get_control_frame_transform(self) -> np.ndarray:
        return self._get_site_transform_by_id(self._control_frame_site_id)

    def _get_control_frame_jacobian(self) -> np.ndarray:
        return self._get_site_local_jacobian(self._control_frame_site_id)

    def _get_control_frame_twist(self) -> np.ndarray:
        return self._get_site_local_twist(self._control_frame_site_id)

    def _get_control_wrench_local_raw(self) -> np.ndarray:
        if not hasattr(self, "_control_force_slice"):
            force_sensor_id = self.model.sensor("attachment_site_force_sensor").id
            torque_sensor_id = self.model.sensor("attachment_site_torque_sensor").id
            force_adr = int(self.model.sensor_adr[force_sensor_id])
            torque_adr = int(self.model.sensor_adr[torque_sensor_id])
            self._control_force_slice = slice(force_adr, force_adr + 3)
            self._control_torque_slice = slice(torque_adr, torque_adr + 3)
        force = np.asarray(
            self.data.sensordata[self._control_force_slice],
            dtype=np.float64,
        ).copy()
        torque = np.asarray(
            self.data.sensordata[self._control_torque_slice],
            dtype=np.float64,
        ).copy()
        # MuJoCo site force/torque sensors report the child-parent interaction
        # wrench. 这里翻转为“环境作用在末端上的外力”语义，与参考实现一致。
        return -np.concatenate([force, torque])

    def _tare_control_wrench(self) -> None:
        self._control_wrench_bias_local = self._get_control_wrench_local_raw()

    def _get_control_wrench_local(self) -> np.ndarray:
        return self._get_control_wrench_local_raw() - self._control_wrench_bias_local

    def _get_control_frame_wrench(self) -> np.ndarray:
        return self._get_control_wrench_local()

    def _get_normalized_control_frame_wrench(self) -> np.ndarray:
        wrench_control_body = self._get_control_frame_wrench().astype(np.float64, copy=True)
        force_scale = max(float(self.insertion_cfg.wrench_force_scale), 1e-6)
        torque_scale = max(float(self.insertion_cfg.wrench_torque_scale), 1e-6)
        wrench_control_body[:3] /= force_scale
        wrench_control_body[3:] /= torque_scale
        return np.clip(wrench_control_body, -1.0, 1.0)

    def _get_teleop_control_state(self) -> Dict[str, np.ndarray]:
        return {
            "q": self.data.qpos[self._qpos_addrs].copy(),
            "q_dot": self.data.qvel[self._dof_addrs].copy(),
            "tip_transform": self._get_current_transform(),
            "control_transform": self._get_control_frame_transform(),
            "control_jacobian_body": self._get_control_frame_jacobian(),
            "control_twist_body": self._get_control_frame_twist(),
            "mass_matrix": self._get_arm_mass_matrix(),
            "bias_forces": self._get_arm_bias_forces(),
            "wrench_body": self._get_control_frame_wrench(),
        }

    def _set_joint_torque_command(self, joint_torque_command: np.ndarray) -> None:
        self.data.ctrl[self._actuator_ids] = np.asarray(
            joint_torque_command,
            dtype=np.float64,
        )

    def _body_twist_to_joint_velocity(self, body_twist: np.ndarray) -> np.ndarray:
        jacobian = self._get_site_jacobian()
        damping_matrix = self.insertion_cfg.ik_damping * np.eye(6, dtype=np.float64)
        jjt = jacobian @ jacobian.T
        jacobian_pinv = jacobian.T @ np.linalg.solve(
            jjt + damping_matrix,
            np.eye(6, dtype=np.float64),
        )
        qdot = jacobian_pinv @ np.asarray(body_twist, dtype=np.float64).reshape((6,))
        return np.clip(
            qdot,
            -np.asarray(self.insertion_cfg.joint_velocity_limits, dtype=np.float64),
            np.asarray(self.insertion_cfg.joint_velocity_limits, dtype=np.float64),
        )

    def _sync_controller_reference_to_current_state(self) -> None:
        self._teleop_runtime.reset_reference()

    def reanchor_teleop_target_to_current_pose(self) -> None:
        """将 teleop 目标重新锚定到当前真实末端位姿。"""
        self._tare_control_wrench()
        self._sync_controller_reference_to_current_state()

    def _get_controller_target_pose(self) -> np.ndarray:
        target_transform = self._teleop_runtime.get_target_tip_transform()
        if target_transform is None:
            target_transform = self._get_current_transform()
        target_quat = Rotation.from_matrix(target_transform[:3, :3]).as_quat()
        return np.concatenate([target_transform[:3, 3], target_quat])

    def set_controller_target_to_goal_tip(self) -> None:
        """Set the impedance target directly to the current goal tip pose."""
        controller_state = self._teleop_runtime.get_internal_state()
        controller_state["target_tip_transform"] = self._goal_tip_transform.copy()
        controller_state["filtered_command"] = np.zeros(6, dtype=np.float64)
        self._teleop_runtime.set_internal_state(controller_state)

    def _compute_rotation_error(self, tip_rot: np.ndarray) -> float:
        rotation_delta = self._goal_tip_transform[:3, :3].T @ tip_rot
        return float(np.linalg.norm(log_so3(rotation_delta)))

    def _is_success(self, tip_pos: np.ndarray, tip_rot: np.ndarray) -> bool:
        cfg = self.insertion_cfg
        tip_lin_vel, tip_ang_vel = self._get_ee_velocity()
        xy_error, z_gap, rot_error, _ = self._compute_goal_frame_errors(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )
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
        _, tip_rot = self._get_ee_pose()
        xy_error, z_gap, _, _ = self._compute_goal_frame_errors(
            tip_pos=tip_pos,
            tip_rot=tip_rot,
        )
        obvious_misalignment = (
            z_gap < cfg.failure_z_gap_threshold
            and xy_error > cfg.failure_xy_threshold
        )
        return bool(out_of_workspace or obvious_misalignment)

    @staticmethod
    def _body_quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape((4,))
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        return Rotation.from_quat(quat_xyzw).as_matrix()

    def _sample_eval_task_case(self) -> tuple[np.ndarray, float]:
        cfg = self.insertion_cfg
        xy_offset = self.np_random.uniform(cfg.eval_case_xy_low, cfg.eval_case_xy_high)
        yaw_offset = float(
            self.np_random.uniform(-cfg.eval_case_yaw_range, cfg.eval_case_yaw_range)
        )
        return np.asarray(xy_offset, dtype=np.float64), yaw_offset

    def _restore_default_task_case(self) -> None:
        self.model.body_pos[self._socket_body_id] = self._default_socket_body_pos.copy()
        self.model.body_quat[self._socket_body_id] = self._default_socket_body_quat.copy()
        mujoco.mj_forward(self.model, self.data)
        self._current_task_case_xy[:] = 0.0
        self._current_task_case_yaw = 0.0
        self._goal_tip_transform = self._default_goal_tip_transform.copy()
        self._preinsert_tip_transform = self._default_preinsert_tip_transform.copy()
        self._update_task_geometry_state()

    def _apply_task_case(self, *, xy_offset: np.ndarray, yaw_offset: float) -> None:
        case_body_rotation = self._default_socket_body_rot @ exp_so3(
            np.array([0.0, 0.0, yaw_offset], dtype=np.float64)
        )
        body_position = self._default_socket_body_pos.copy()
        body_position[:2] += np.asarray(xy_offset, dtype=np.float64).reshape((2,))
        self.model.body_pos[self._socket_body_id] = body_position
        self.model.body_quat[self._socket_body_id] = rotation_matrix_to_quat_wxyz(
            case_body_rotation
        )
        mujoco.mj_forward(self.model, self.data)

        task_pos, task_rot = self._get_task_frame_pose()
        goal_rotation = task_rot @ self._default_task_to_goal_rot
        self._goal_tip_transform = make_transform(
            rotation=goal_rotation,
            translation=task_pos,
        )
        self._preinsert_tip_transform = self._default_preinsert_tip_transform.copy()
        self._current_task_case_xy = np.asarray(xy_offset, dtype=np.float64).reshape((2,))
        self._current_task_case_yaw = float(yaw_offset)
        self._update_task_geometry_state()

    def _configure_task_case(self, options: Optional[Dict[str, Any]]) -> None:
        self._restore_default_task_case()
        options = options or {}
        task_case = options.get("task_case")
        if isinstance(task_case, dict):
            xy_offset = np.asarray(
                task_case.get("xy", np.zeros(2, dtype=np.float64)),
                dtype=np.float64,
            ).reshape((2,))
            yaw_offset = float(task_case.get("yaw", 0.0))
            self._apply_task_case(xy_offset=xy_offset, yaw_offset=yaw_offset)
            return
        if bool(options.get("sample_task_case", False)):
            xy_offset, yaw_offset = self._sample_eval_task_case()
            self._apply_task_case(xy_offset=xy_offset, yaw_offset=yaw_offset)

    def _update_task_geometry_state(self) -> None:
        self._socket_pos, self._socket_rot = self._get_socket_pose()
        goal_to_preinsert = compose_transforms(
            invert_transform(self._goal_tip_transform),
            self._preinsert_tip_transform,
        )
        self._goal_frame_preinsert_translation = goal_to_preinsert[:3, 3].copy()
        axial_reference = float(self._goal_frame_preinsert_translation[2])
        self._axial_reference_sign = np.sign(axial_reference) if abs(axial_reference) > 1e-9 else 1.0
        self._max_insertion_depth = max(abs(axial_reference), 1e-6)
        self._preinsert_tip_pos = self._preinsert_tip_transform[:3, 3].copy()

    def _compute_pose_error_body(
        self,
        *,
        tip_pos: np.ndarray,
        tip_rot: np.ndarray,
    ) -> np.ndarray:
        tip_transform = make_transform(rotation=tip_rot, translation=tip_pos)
        relative = compose_transforms(invert_transform(tip_transform), self._goal_tip_transform)
        return log_se3(relative)

    def _get_ee_body_twist(self) -> np.ndarray:
        ee_lin_vel, ee_ang_vel = self._get_ee_velocity()
        _, ee_rot = self._get_ee_pose()
        rotation_world_to_body = ee_rot.T
        linear_body = rotation_world_to_body @ ee_lin_vel
        angular_body = rotation_world_to_body @ ee_ang_vel
        return np.concatenate([linear_body, angular_body])

    def _compute_goal_frame_errors(
        self,
        *,
        tip_pos: np.ndarray,
        tip_rot: np.ndarray,
    ) -> tuple[float, float, float, float]:
        goal_transform = self._goal_tip_transform
        goal_rot = goal_transform[:3, :3]
        goal_pos = goal_transform[:3, 3]
        tip_translation_goal = goal_rot.T @ (tip_pos - goal_pos)
        axial_offset = float(tip_translation_goal[2] * self._axial_reference_sign)
        xy_error = float(np.linalg.norm(tip_translation_goal[:2]))
        z_gap = float(max(axial_offset, 0.0))
        rot_error = float(np.linalg.norm(log_so3(goal_rot.T @ tip_rot)))
        insertion_depth = float(
            np.clip(
                self._max_insertion_depth - axial_offset,
                0.0,
                self._max_insertion_depth,
            )
        )
        return xy_error, z_gap, rot_error, insertion_depth

    def capture_diagnostic_snapshot(self) -> Dict[str, Any]:
        """Capture enough simulator/controller state to reproduce one state."""
        tip_pos, tip_rot = self._get_ee_pose()
        tip_transform = make_transform(rotation=tip_rot, translation=tip_pos)
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "step_count": int(self._step_count),
            "task_case_xy": self._current_task_case_xy.copy(),
            "task_case_yaw": float(self._current_task_case_yaw),
            "reset_case_pos_noise": self._current_reset_case_pos_noise.copy(),
            "reset_case_yaw": float(self._current_reset_case_yaw),
            "goal_tip_transform": self._goal_tip_transform.copy(),
            "preinsert_tip_transform": self._preinsert_tip_transform.copy(),
            "tip_transform": tip_transform,
            "twist_body": self._get_ee_body_twist().copy(),
            "control_wrench_bias_local": self._control_wrench_bias_local.copy(),
            "controller_state": self._teleop_runtime.get_internal_state(),
        }

    def restore_diagnostic_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore a snapshot captured by :meth:`capture_diagnostic_snapshot`."""
        self._configure_task_case(
            {
                "task_case": {
                    "xy": snapshot["task_case_xy"],
                    "yaw": snapshot["task_case_yaw"],
                }
            }
        )
        self.data.qpos[:] = np.asarray(snapshot["qpos"], dtype=np.float64)
        self.data.qvel[:] = np.asarray(snapshot["qvel"], dtype=np.float64)
        self.data.ctrl[:] = np.asarray(snapshot["ctrl"], dtype=np.float64)
        self._step_count = int(snapshot["step_count"])
        if "reset_case_pos_noise" in snapshot:
            self._current_reset_case_pos_noise = np.asarray(
                snapshot["reset_case_pos_noise"],
                dtype=np.float64,
            ).copy()
        if "reset_case_yaw" in snapshot:
            self._current_reset_case_yaw = float(snapshot["reset_case_yaw"])
        mujoco.mj_forward(self.model, self.data)
        if "control_wrench_bias_local" in snapshot:
            self._control_wrench_bias_local = np.asarray(
                snapshot["control_wrench_bias_local"],
                dtype=np.float64,
            ).copy()
        self._teleop_runtime.set_internal_state(snapshot["controller_state"])

    def restore_transformed_diagnostic_snapshot(
        self,
        snapshot: Dict[str, Any],
        *,
        xy_offset: np.ndarray,
        yaw_offset: float,
    ) -> None:
        """Restore a task-case transformed version of a captured snapshot.

        The matched state keeps the same task-relative pose error by applying the
        rigid task-case transform to the robot tip pose and the controller's
        internal target/virtual transforms. Body-frame twist is preserved by
        solving a local Jacobian least-squares velocity after IK.
        """
        self._configure_task_case(
            {
                "task_case": {
                    "xy": np.asarray(xy_offset, dtype=np.float64).reshape((2,)),
                    "yaw": float(yaw_offset),
                }
            }
        )

        delta_transform = compose_transforms(
            self._goal_tip_transform,
            invert_transform(np.asarray(snapshot["goal_tip_transform"], dtype=np.float64)),
        )
        matched_tip_transform = compose_transforms(
            delta_transform,
            np.asarray(snapshot["tip_transform"], dtype=np.float64),
        )
        self._set_tip_pose_with_ik(
            matched_tip_transform[:3, 3],
            matched_tip_transform[:3, :3],
        )
        self.data.qvel[:] = 0.0
        self.data.qvel[self._dof_addrs] = self._body_twist_to_joint_velocity(
            snapshot["twist_body"]
        )
        self._sync_actuator_ctrl_to_current_state()
        mujoco.mj_forward(self.model, self.data)

        controller_state = snapshot["controller_state"].copy()
        if controller_state.get("target_tip_transform") is not None:
            controller_state["target_tip_transform"] = compose_transforms(
                delta_transform,
                np.asarray(controller_state["target_tip_transform"], dtype=np.float64),
            )
        nric_state = controller_state.get("nric_state")
        if isinstance(nric_state, dict):
            if "nominal_q" in nric_state:
                nric_state["nominal_q"] = self.data.qpos[self._qpos_addrs].copy()
            if "nominal_qdot" in nric_state:
                nric_state["nominal_qdot"] = self.data.qvel[self._dof_addrs].copy()
        self._teleop_runtime.set_internal_state(controller_state)
        self._step_count = int(snapshot["step_count"])
