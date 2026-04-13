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

from robot_controllers_mujoco.control.task_space_velocity_controller import (
    TaskSpaceVelocityController,
)
from robot_controllers_mujoco.control.teleop_runtime import TeleopRuntime
from robot_controllers_mujoco.envs.franka_mujoco_env import (
    FrankaMujocoConfig,
    FrankaMujocoEnv,
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

    # Damped least-squares IK used during reset randomisation.
    reset_ik_max_iters: int = 80
    reset_ik_step_scale: float = 0.6
    reset_ik_tolerance: float = 1e-4
    reset_ik_damping: float = 1e-4

    # Reward weights.
    reward_xy_weight: float = 8.0
    reward_z_weight: float = 1.5
    reward_rot_weight: float = 0.3
    reward_depth_weight: float = 2.0
    reward_success_bonus: float = 25.0

    # Success thresholds.
    success_xy_threshold: float = 0.004
    success_rot_threshold: float = 0.12
    success_tip_z_threshold: float = 0.105

    # Failure / workspace bounds.
    workspace_low: np.ndarray = field(
        default_factory=lambda: np.array([0.49, -0.05, 0.05])
    )
    workspace_high: np.ndarray = field(
        default_factory=lambda: np.array([0.62, 0.05, 0.58])
    )
    failure_xy_threshold: float = 0.03
    failure_tip_z_threshold: float = 0.075

    max_episode_steps: int = 200

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

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addrs] = self.cfg.home_qpos.copy()
        self.data.ctrl[self._actuator_ids] = self.cfg.home_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

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
        normalized_depth = float(
            np.clip(
                (self._preinsert_tip_pos[2] - tip_pos[2]) / self._max_insertion_depth,
                0.0,
                1.0,
            )
        )

        reward = (
            -cfg.reward_xy_weight * xy_error
            - cfg.reward_z_weight * z_gap
            - cfg.reward_rot_weight * rot_error
            + cfg.reward_depth_weight * normalized_depth
        )

        if self._is_success(tip_pos, tip_rot):
            reward += cfg.reward_success_bonus

        return float(reward)

    def _check_termination(self, obs: Dict) -> bool:
        tip_pos, tip_rot = self._get_ee_pose()
        return bool(self._is_success(tip_pos, tip_rot) or self._is_failure(tip_pos))

    def _get_info(self, obs: Dict) -> Dict[str, Any]:
        tip_pos, tip_rot = self._get_ee_pose()
        socket_pos, _ = self._get_socket_pose()

        xy_error = float(np.linalg.norm(tip_pos[:2] - socket_pos[:2]))
        z_gap = float(max(tip_pos[2] - socket_pos[2], 0.0))
        rot_error = self._compute_rotation_error(tip_rot)
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
            "insertion_depth": insertion_depth,
            "success": success,
            "failure": failure,
        }

    def _sample_reset_tip_pose(self) -> Tuple[np.ndarray, np.ndarray]:
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
        cfg = self.insertion_cfg
        q = self.data.qpos[self._qpos_addrs].copy()

        for _ in range(cfg.reset_ik_max_iters):
            current_pos, current_rot = self._get_ee_pose()
            pos_error = target_tip_pos - current_pos
            rot_error = Rotation.from_matrix(
                target_tip_rot @ current_rot.T
            ).as_rotvec()
            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(error) < cfg.reset_ik_tolerance:
                break

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
            jacobian = np.vstack([
                jacp[:, self._dof_addrs],
                jacr[:, self._dof_addrs],
            ])

            jjt = jacobian @ jacobian.T + cfg.reset_ik_damping * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(jjt, error)
            q = np.clip(
                q + cfg.reset_ik_step_scale * dq,
                self._q_low,
                self._q_high,
            )

            self.data.qpos[self._qpos_addrs] = q
            self.data.ctrl[self._actuator_ids] = q
            mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[self._actuator_ids] = self.data.qpos[self._qpos_addrs].copy()
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
        xy_error = np.linalg.norm(tip_pos[:2] - self._socket_pos[:2])
        rot_error = self._compute_rotation_error(tip_rot)
        return bool(
            xy_error < cfg.success_xy_threshold
            and rot_error < cfg.success_rot_threshold
            and tip_pos[2] <= cfg.success_tip_z_threshold
        )

    def _is_failure(self, tip_pos: np.ndarray) -> bool:
        cfg = self.insertion_cfg
        out_of_workspace = np.any(tip_pos < cfg.workspace_low) or np.any(
            tip_pos > cfg.workspace_high
        )
        xy_error = np.linalg.norm(tip_pos[:2] - self._socket_pos[:2])
        obvious_misalignment = (
            tip_pos[2] <= cfg.failure_tip_z_threshold
            and xy_error > cfg.failure_xy_threshold
        )
        return bool(out_of_workspace or obvious_misalignment)
