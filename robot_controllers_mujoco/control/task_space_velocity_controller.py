"""
Shared task-space velocity controller used by both teleop and MuJoCo envs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class TaskSpaceVelocityController:
    """Stateful task-space controller with optional world-frame admittance."""

    def __init__(
        self,
        *,
        input_deadzone: float,
        input_filter_alpha: float,
        max_reference_linear_velocity: float,
        max_reference_angular_velocity: float,
        pose_error_gain: float,
        ik_damping: float,
        nullspace_gain: float,
        joint_velocity_limits: np.ndarray,
        joint_limit_margin: float,
        lower_position_limits: np.ndarray,
        upper_position_limits: np.ndarray,
        enable_admittance: bool = False,
        wrench_filter_alpha: float = 0.2,
        force_deadband: Optional[np.ndarray] = None,
        torque_deadband: Optional[np.ndarray] = None,
        max_force_world: Optional[np.ndarray] = None,
        max_torque_world: Optional[np.ndarray] = None,
        admittance_mass: Optional[np.ndarray] = None,
        admittance_damping: Optional[np.ndarray] = None,
        admittance_stiffness: Optional[np.ndarray] = None,
        max_compliance_linear_offset: float = 0.02,
        max_compliance_angular_offset: float = 0.2,
        max_compliance_linear_velocity: float = 0.10,
        max_compliance_angular_velocity: float = 0.5,
        joint_velocity_filter_alpha: float = 1.0,
    ) -> None:
        self.input_deadzone = float(input_deadzone)
        self.input_filter_alpha = float(np.clip(input_filter_alpha, 0.0, 1.0))
        self.max_reference_linear_velocity = float(max_reference_linear_velocity)
        self.max_reference_angular_velocity = float(max_reference_angular_velocity)
        self.pose_error_gain = float(pose_error_gain)
        self.ik_damping = float(ik_damping)
        self.nullspace_gain = float(nullspace_gain)
        self.joint_velocity_limits = np.asarray(
            joint_velocity_limits,
            dtype=np.float64,
        ).copy()
        self.joint_limit_margin = float(joint_limit_margin)
        self.lower_position_limits = np.asarray(
            lower_position_limits,
            dtype=np.float64,
        ).copy()
        self.upper_position_limits = np.asarray(
            upper_position_limits,
            dtype=np.float64,
        ).copy()

        self.enable_admittance = bool(enable_admittance)
        self.wrench_filter_alpha = float(np.clip(wrench_filter_alpha, 0.0, 1.0))
        self.force_deadband = self._coerce_vector(force_deadband, size=3, fill=0.0)
        self.torque_deadband = self._coerce_vector(torque_deadband, size=3, fill=0.0)
        self.max_force_world = self._coerce_vector(
            max_force_world,
            size=3,
            fill=np.inf,
        )
        self.max_torque_world = self._coerce_vector(
            max_torque_world,
            size=3,
            fill=np.inf,
        )
        self.admittance_mass = np.maximum(
            self._coerce_vector(admittance_mass, size=6, fill=1.0),
            1e-6,
        )
        self.admittance_damping = self._coerce_vector(
            admittance_damping,
            size=6,
            fill=0.0,
        )
        self.admittance_stiffness = self._coerce_vector(
            admittance_stiffness,
            size=6,
            fill=0.0,
        )
        self.max_compliance_linear_offset = float(max_compliance_linear_offset)
        self.max_compliance_angular_offset = float(max_compliance_angular_offset)
        self.max_compliance_linear_velocity = float(max_compliance_linear_velocity)
        self.max_compliance_angular_velocity = float(max_compliance_angular_velocity)
        self.joint_velocity_filter_alpha = float(
            np.clip(joint_velocity_filter_alpha, 0.0, 1.0)
        )

        self.filtered_command = np.zeros(6, dtype=np.float64)
        self.filtered_wrench_world = np.zeros(6, dtype=np.float64)
        self.compliance_offset_world = np.zeros(6, dtype=np.float64)
        self.compliance_twist_world = np.zeros(6, dtype=np.float64)
        self.filtered_joint_velocity_command = np.zeros_like(
            self.joint_velocity_limits,
            dtype=np.float64,
        )
        self.target_transform: Optional[np.ndarray] = None
        self.virtual_transform: Optional[np.ndarray] = None
        self.posture_reference: Optional[np.ndarray] = None

    @staticmethod
    def _coerce_vector(
        value: Optional[np.ndarray],
        *,
        size: int,
        fill: float,
    ) -> np.ndarray:
        if value is None:
            return np.full((size,), fill, dtype=np.float64)

        array = np.asarray(value, dtype=np.float64).reshape((-1,))
        if array.shape != (size,):
            raise ValueError(f"Expected vector shape ({size},), got {array.shape}")
        return array.copy()

    def reset(
        self,
        *,
        current_transform: np.ndarray,
        current_q: np.ndarray,
        posture_reference: Optional[np.ndarray] = None,
    ) -> None:
        """Reset controller state to the current robot pose."""
        current_transform = np.asarray(current_transform, dtype=np.float64)
        self.target_transform = current_transform.copy()
        self.virtual_transform = current_transform.copy()
        if posture_reference is None:
            self.posture_reference = np.asarray(current_q, dtype=np.float64).copy()
        else:
            self.posture_reference = np.asarray(
                posture_reference,
                dtype=np.float64,
            ).copy()

        self.filtered_command[:] = 0.0
        self.filtered_wrench_world[:] = 0.0
        self.compliance_offset_world[:] = 0.0
        self.compliance_twist_world[:] = 0.0
        self.filtered_joint_velocity_command[:] = 0.0

    def get_target_transform(self) -> Optional[np.ndarray]:
        """Return a copy of the currently integrated target pose."""
        if self.target_transform is None:
            return None
        return self.target_transform.copy()

    def get_virtual_transform(self) -> Optional[np.ndarray]:
        """Return a copy of the world-frame admittance-compensated target pose."""
        if self.virtual_transform is None:
            return None
        return self.virtual_transform.copy()

    def compute_command(
        self,
        *,
        raw_action: np.ndarray,
        current_transform: np.ndarray,
        jacobian: np.ndarray,
        current_q: np.ndarray,
        dt: float,
        wrench_world: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """根据归一化 6D 输入计算关节速度命令。

        控制链路为：输入滤波、积分名义任务空间目标、可选叠加导纳虚拟目标、
        计算 LOCAL 位姿误差，然后求解速度级阻尼最小二乘问题。这里不是环境
        reset 时使用的非线性位姿 IK。
        """
        if self.target_transform is None or self.posture_reference is None:
            self.reset(
                current_transform=current_transform,
                current_q=current_q,
            )

        filtered_command = self._filter_input(np.asarray(raw_action, dtype=np.float64))
        self._integrate_target_transform(filtered_command, dt=float(dt))

        tracking_transform = self.target_transform
        if self.enable_admittance:
            filtered_wrench_world = self._filter_wrench_world(wrench_world)
            self._update_virtual_transform(
                dt=float(dt),
                wrench_world=filtered_wrench_world,
            )
            tracking_transform = self.virtual_transform
        else:
            self.virtual_transform = self.target_transform.copy()

        pose_error = self._compute_local_pose_error(
            current_transform=np.asarray(current_transform, dtype=np.float64),
            target_transform=tracking_transform,
        )
        qdot_command = self._compute_joint_velocity_command(
            current_q=np.asarray(current_q, dtype=np.float64),
            jacobian=np.asarray(jacobian, dtype=np.float64),
            pose_error=pose_error,
        )
        return self._filter_joint_velocity_command(qdot_command)

    def _filter_input(self, raw_action: np.ndarray) -> np.ndarray:
        if raw_action.shape != (6,):
            raise ValueError(
                f"Task-space input dimension mismatch: expected (6,), got {raw_action.shape}"
            )

        action = raw_action.copy()
        action[np.abs(action) < self.input_deadzone] = 0.0
        if not np.any(action):
            self.filtered_command[:] = 0.0
            return np.zeros(6, dtype=np.float64)

        self.filtered_command = (
            self.input_filter_alpha * action
            + (1.0 - self.input_filter_alpha) * self.filtered_command
        )
        return self.filtered_command.copy()

    def _integrate_target_transform(
        self,
        filtered_command: np.ndarray,
        dt: float,
    ) -> None:
        """将操作者或 policy 命令积分到名义目标位姿。

        当前 teleop 约定是有意混合的：平移命令按世界坐标系解释，旋转增量在
        目标 LOCAL 坐标系下右乘，以保留 SpaceMouse 的操作手感。
        """
        if self.target_transform is None:
            raise RuntimeError("Task-space target transform is not initialized.")

        delta_translation_world = (
            filtered_command[:3] * dt * self.max_reference_linear_velocity
        )
        delta_rotation_local = (
            filtered_command[3:] * dt * self.max_reference_angular_velocity
        )

        target_transform = self.target_transform.copy()
        target_rotation = target_transform[:3, :3]
        target_position = target_transform[:3, 3]

        target_position = target_position + delta_translation_world
        target_rotation = target_rotation @ Rotation.from_rotvec(
            delta_rotation_local
        ).as_matrix()
        target_rotation = self._project_rotation_matrix(target_rotation)

        self.target_transform[:3, :3] = target_rotation
        self.target_transform[:3, 3] = target_position

    def _filter_wrench_world(self, wrench_world: Optional[np.ndarray]) -> np.ndarray:
        if wrench_world is None:
            filtered_input = np.zeros(6, dtype=np.float64)
        else:
            filtered_input = np.asarray(wrench_world, dtype=np.float64).reshape((-1,))
            if filtered_input.shape != (6,):
                raise ValueError(
                    f"Expected world-frame wrench shape (6,), got {filtered_input.shape}"
                )

        filtered_input = filtered_input.copy()
        filtered_input[:3][np.abs(filtered_input[:3]) < self.force_deadband] = 0.0
        filtered_input[3:][np.abs(filtered_input[3:]) < self.torque_deadband] = 0.0
        filtered_input[:3] = np.clip(
            filtered_input[:3],
            -self.max_force_world,
            self.max_force_world,
        )
        filtered_input[3:] = np.clip(
            filtered_input[3:],
            -self.max_torque_world,
            self.max_torque_world,
        )
        self.filtered_wrench_world = (
            self.wrench_filter_alpha * filtered_input
            + (1.0 - self.wrench_filter_alpha) * self.filtered_wrench_world
        )
        return self.filtered_wrench_world.copy()

    def _update_virtual_transform(
        self,
        *,
        dt: float,
        wrench_world: np.ndarray,
    ) -> None:
        """更新可选的导纳目标偏置。

        存储的柔顺状态是相对名义目标的偏置，因此 wrench 输入消失后会衰减回
        零，而不是永久漂移操作者或 policy 的参考目标。
        """
        if self.target_transform is None:
            raise RuntimeError("Task-space target transform is not initialized.")

        acceleration_world = (
            wrench_world
            - self.admittance_damping * self.compliance_twist_world
            - self.admittance_stiffness * self.compliance_offset_world
        ) / self.admittance_mass

        self.compliance_twist_world = (
            self.compliance_twist_world + acceleration_world * dt
        )
        self.compliance_twist_world[:3] = np.clip(
            self.compliance_twist_world[:3],
            -self.max_compliance_linear_velocity,
            self.max_compliance_linear_velocity,
        )
        self.compliance_twist_world[3:] = np.clip(
            self.compliance_twist_world[3:],
            -self.max_compliance_angular_velocity,
            self.max_compliance_angular_velocity,
        )

        self.compliance_offset_world = (
            self.compliance_offset_world + self.compliance_twist_world * dt
        )
        self.compliance_offset_world[:3] = np.clip(
            self.compliance_offset_world[:3],
            -self.max_compliance_linear_offset,
            self.max_compliance_linear_offset,
        )
        self.compliance_offset_world[3:] = np.clip(
            self.compliance_offset_world[3:],
            -self.max_compliance_angular_offset,
            self.max_compliance_angular_offset,
        )

        target_rotation = self.target_transform[:3, :3]
        target_position = self.target_transform[:3, 3]
        virtual_rotation = (
            Rotation.from_rotvec(self.compliance_offset_world[3:]).as_matrix()
            @ target_rotation
        )

        self.virtual_transform = np.eye(4, dtype=np.float64)
        self.virtual_transform[:3, :3] = self._project_rotation_matrix(virtual_rotation)
        self.virtual_transform[:3, 3] = (
            target_position + self.compliance_offset_world[:3]
        )

    @staticmethod
    def _project_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
        u, _, vh = np.linalg.svd(rotation_matrix)
        projected = u @ vh
        if np.linalg.det(projected) < 0.0:
            u[:, -1] *= -1.0
            projected = u @ vh
        return projected

    def _compute_local_pose_error(
        self,
        *,
        current_transform: np.ndarray,
        target_transform: np.ndarray,
    ) -> np.ndarray:
        current_rotation = current_transform[:3, :3]
        current_position = current_transform[:3, 3]
        target_rotation = target_transform[:3, :3]
        target_position = target_transform[:3, 3]

        position_error_local = current_rotation.T @ (target_position - current_position)
        rotation_error_local = Rotation.from_matrix(
            current_rotation.T @ target_rotation
        ).as_rotvec()
        return np.concatenate([position_error_local, rotation_error_local])

    def _compute_joint_velocity_command(
        self,
        *,
        current_q: np.ndarray,
        jacobian: np.ndarray,
        pose_error: np.ndarray,
    ) -> np.ndarray:
        """求解速度级 DLS 任务，并叠加零空间姿态控制。

        该方法输出单个控制周期的 qdot。它刻意不求解全局位姿 IK，也不会直接
        重写关节位置。
        """
        damping_matrix = self.ik_damping * np.eye(6, dtype=np.float64)
        jjt = jacobian @ jacobian.T
        jacobian_pinv = jacobian.T @ np.linalg.solve(
            jjt + damping_matrix,
            np.eye(6, dtype=np.float64),
        )
        qdot_task = self.pose_error_gain * (jacobian_pinv @ pose_error)

        qdot_null = self.nullspace_gain * (self.posture_reference - current_q)
        nullspace_projector = (
            np.eye(jacobian.shape[1], dtype=np.float64) - jacobian_pinv @ jacobian
        )
        qdot_command = qdot_task + nullspace_projector @ qdot_null
        qdot_command = self._apply_joint_limit_guard(current_q, qdot_command)
        qdot_command = np.clip(
            qdot_command,
            -self.joint_velocity_limits,
            self.joint_velocity_limits,
        )
        return qdot_command

    def _filter_joint_velocity_command(
        self,
        qdot_command: np.ndarray,
    ) -> np.ndarray:
        self.filtered_joint_velocity_command = (
            self.joint_velocity_filter_alpha * qdot_command
            + (1.0 - self.joint_velocity_filter_alpha)
            * self.filtered_joint_velocity_command
        )
        return self.filtered_joint_velocity_command.copy()

    def _apply_joint_limit_guard(
        self,
        current_q: np.ndarray,
        qdot_command: np.ndarray,
    ) -> np.ndarray:
        """阻断会让关节继续朝限位外运动的速度分量。"""
        qdot_safe = np.asarray(qdot_command, dtype=np.float64).copy()
        finite_mask = np.isfinite(self.lower_position_limits) & np.isfinite(
            self.upper_position_limits
        )

        lower_block = (
            finite_mask
            & (current_q <= self.lower_position_limits + self.joint_limit_margin)
            & (qdot_safe < 0.0)
        )
        upper_block = (
            finite_mask
            & (current_q >= self.upper_position_limits - self.joint_limit_margin)
            & (qdot_safe > 0.0)
        )
        qdot_safe[lower_block | upper_block] = 0.0
        return qdot_safe
