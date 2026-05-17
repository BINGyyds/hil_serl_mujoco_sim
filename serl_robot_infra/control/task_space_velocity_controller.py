"""
Shared task-space velocity controller used by both teleop and MuJoCo envs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from serl_robot_infra.common.se3_utils import (
    compose_transforms,
    exp_se3,
    exp_so3,
    log_se3,
    log_so3,
    make_transform,
    project_rotation_matrix,
)


class TaskSpaceVelocityController:
    """带可选几何导纳层的任务空间速度控制器。"""

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
        # 配置名沿用 max_torque_world 以兼容历史 override；几何导纳内部按
        # tool-tip 本体坐标系 torque 使用。
        self.max_torque_body = self._coerce_vector(
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
        self.filtered_force_world = np.zeros(3, dtype=np.float64)
        self.filtered_torque_body = np.zeros(3, dtype=np.float64)
        self.compliance_position_world = np.zeros(3, dtype=np.float64)
        self.compliance_linear_velocity_world = np.zeros(3, dtype=np.float64)
        self.compliance_rotation_body = np.zeros(3, dtype=np.float64)
        self.compliance_angular_velocity_body = np.zeros(3, dtype=np.float64)
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
        """将控制器内部状态重置到当前机器人位姿。"""
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
        self.filtered_force_world[:] = 0.0
        self.filtered_torque_body[:] = 0.0
        self.compliance_position_world[:] = 0.0
        self.compliance_linear_velocity_world[:] = 0.0
        self.compliance_rotation_body[:] = 0.0
        self.compliance_angular_velocity_body[:] = 0.0
        self.filtered_joint_velocity_command[:] = 0.0

    def get_target_transform(self) -> Optional[np.ndarray]:
        """返回当前积分得到的名义目标位姿副本。"""
        if self.target_transform is None:
            return None
        return self.target_transform.copy()

    def get_virtual_transform(self) -> Optional[np.ndarray]:
        """返回导纳补偿后的虚拟目标位姿副本。"""
        if self.virtual_transform is None:
            return None
        return self.virtual_transform.copy()

    def get_internal_state(self) -> dict[str, Optional[np.ndarray]]:
        """Return a copy of controller state for diagnostics/reproducibility."""
        return {
            "filtered_command": self.filtered_command.copy(),
            "filtered_force_world": self.filtered_force_world.copy(),
            "filtered_torque_body": self.filtered_torque_body.copy(),
            "compliance_position_world": self.compliance_position_world.copy(),
            "compliance_linear_velocity_world": (
                self.compliance_linear_velocity_world.copy()
            ),
            "compliance_rotation_body": self.compliance_rotation_body.copy(),
            "compliance_angular_velocity_body": (
                self.compliance_angular_velocity_body.copy()
            ),
            "filtered_joint_velocity_command": (
                self.filtered_joint_velocity_command.copy()
            ),
            "target_transform": None
            if self.target_transform is None
            else self.target_transform.copy(),
            "virtual_transform": None
            if self.virtual_transform is None
            else self.virtual_transform.copy(),
            "posture_reference": None
            if self.posture_reference is None
            else self.posture_reference.copy(),
        }

    def set_internal_state(self, state: dict[str, Optional[np.ndarray]]) -> None:
        """Restore controller state captured by :meth:`get_internal_state`."""
        self.filtered_command = np.asarray(
            state["filtered_command"],
            dtype=np.float64,
        ).copy()
        self.filtered_force_world = np.asarray(
            state["filtered_force_world"],
            dtype=np.float64,
        ).copy()
        self.filtered_torque_body = np.asarray(
            state["filtered_torque_body"],
            dtype=np.float64,
        ).copy()
        self.compliance_position_world = np.asarray(
            state["compliance_position_world"],
            dtype=np.float64,
        ).copy()
        self.compliance_linear_velocity_world = np.asarray(
            state["compliance_linear_velocity_world"],
            dtype=np.float64,
        ).copy()
        self.compliance_rotation_body = np.asarray(
            state["compliance_rotation_body"],
            dtype=np.float64,
        ).copy()
        self.compliance_angular_velocity_body = np.asarray(
            state["compliance_angular_velocity_body"],
            dtype=np.float64,
        ).copy()
        self.filtered_joint_velocity_command = np.asarray(
            state["filtered_joint_velocity_command"],
            dtype=np.float64,
        ).copy()
        self.target_transform = None
        if state.get("target_transform") is not None:
            self.target_transform = np.asarray(
                state["target_transform"],
                dtype=np.float64,
            ).copy()
        self.virtual_transform = None
        if state.get("virtual_transform") is not None:
            self.virtual_transform = np.asarray(
                state["virtual_transform"],
                dtype=np.float64,
            ).copy()
        self.posture_reference = None
        if state.get("posture_reference") is not None:
            self.posture_reference = np.asarray(
                state["posture_reference"],
                dtype=np.float64,
            ).copy()

    def compute_command(
        self,
        *,
        raw_action: np.ndarray,
        current_transform: np.ndarray,
        jacobian: np.ndarray,
        current_q: np.ndarray,
        dt: float,
        wrench_local: Optional[np.ndarray] = None,
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
            force_world, torque_body = self._filter_wrench_local(
                wrench_local=wrench_local,
                current_rotation=np.asarray(current_transform, dtype=np.float64)[:3, :3],
            )
            self._update_virtual_transform(
                dt=float(dt),
                force_world=force_world,
                torque_body=torque_body,
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
        """将 body-frame 6D twist 命令积分到名义目标位姿。"""
        if self.target_transform is None:
            raise RuntimeError("Task-space target transform is not initialized.")

        body_twist = np.zeros(6, dtype=np.float64)
        body_twist[:3] = (
            filtered_command[:3] * dt * self.max_reference_linear_velocity
        )
        body_twist[3:] = (
            filtered_command[3:] * dt * self.max_reference_angular_velocity
        )
        self.target_transform = compose_transforms(
            self.target_transform,
            exp_se3(body_twist),
        )
        self.target_transform[:3, :3] = project_rotation_matrix(
            self.target_transform[:3, :3]
        )

    def _filter_wrench_local(
        self,
        *,
        wrench_local: Optional[np.ndarray],
        current_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """读取本体系 wrench，并分别得到世界系 force 和本体系 torque。"""
        if wrench_local is None:
            force_body = np.zeros(3, dtype=np.float64)
            torque_body = np.zeros(3, dtype=np.float64)
        else:
            wrench = np.asarray(wrench_local, dtype=np.float64).reshape((-1,))
            if wrench.shape != (6,):
                raise ValueError(
                    f"Expected local-frame wrench shape (6,), got {wrench.shape}"
                )
            force_body = wrench[:3].copy()
            torque_body = wrench[3:].copy()

        force_world = current_rotation @ force_body
        force_world[np.abs(force_world) < self.force_deadband] = 0.0
        torque_body[np.abs(torque_body) < self.torque_deadband] = 0.0
        force_world = np.clip(
            force_world,
            -self.max_force_world,
            self.max_force_world,
        )
        torque_body = np.clip(
            torque_body,
            -self.max_torque_body,
            self.max_torque_body,
        )
        self.filtered_force_world = (
            self.wrench_filter_alpha * force_world
            + (1.0 - self.wrench_filter_alpha) * self.filtered_force_world
        )
        self.filtered_torque_body = (
            self.wrench_filter_alpha * torque_body
            + (1.0 - self.wrench_filter_alpha) * self.filtered_torque_body
        )
        return self.filtered_force_world.copy(), self.filtered_torque_body.copy()

    def _update_virtual_transform(
        self,
        *,
        dt: float,
        force_world: np.ndarray,
        torque_body: np.ndarray,
    ) -> None:
        """更新可选的导纳目标偏置。

        平移柔顺在世界坐标系下积分；旋转柔顺采用 body/right-invariant SO(3)
        导纳。两者都是相对名义目标的偏置，wrench 消失后会衰减回零。
        """
        if self.target_transform is None:
            raise RuntimeError("Task-space target transform is not initialized.")

        self._integrate_translation_admittance_rk4(
            dt=dt,
            force_world=force_world,
        )
        self._integrate_rotation_admittance_rk4(
            dt=dt,
            torque_body=torque_body,
        )

        target_rotation = self.target_transform[:3, :3]
        target_position = self.target_transform[:3, 3]
        rotation_offset_body = self._exp_so3(self.compliance_rotation_body)
        virtual_rotation = target_rotation @ rotation_offset_body

        self.virtual_transform = make_transform(
            rotation=virtual_rotation,
            translation=target_position + self.compliance_position_world,
        )

    def _translation_acceleration(
        self,
        position_world: np.ndarray,
        velocity_world: np.ndarray,
        force_world: np.ndarray,
    ) -> np.ndarray:
        mass = self.admittance_mass[:3]
        damping = self.admittance_damping[:3]
        stiffness = self.admittance_stiffness[:3]
        return (
            force_world - damping * velocity_world - stiffness * position_world
        ) / mass

    def _integrate_translation_admittance_rk4(
        self,
        *,
        dt: float,
        force_world: np.ndarray,
    ) -> None:
        """用 RK4 积分世界系平移导纳二阶系统。"""
        h = float(dt)
        p = self.compliance_position_world
        v = self.compliance_linear_velocity_world

        k1_v = self._translation_acceleration(p, v, force_world)
        k1_p = v
        k2_v = self._translation_acceleration(
            p + 0.5 * h * k1_p,
            v + 0.5 * h * k1_v,
            force_world,
        )
        k2_p = v + 0.5 * h * k1_v
        k3_v = self._translation_acceleration(
            p + 0.5 * h * k2_p,
            v + 0.5 * h * k2_v,
            force_world,
        )
        k3_p = v + 0.5 * h * k2_v
        k4_v = self._translation_acceleration(
            p + h * k3_p,
            v + h * k3_v,
            force_world,
        )
        k4_p = v + h * k3_v

        self.compliance_linear_velocity_world = v + h / 6.0 * (
            k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
        )
        self.compliance_linear_velocity_world = np.clip(
            self.compliance_linear_velocity_world,
            -self.max_compliance_linear_velocity,
            self.max_compliance_linear_velocity,
        )
        self.compliance_position_world = p + h / 6.0 * (
            k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
        )
        self.compliance_position_world = np.clip(
            self.compliance_position_world,
            -self.max_compliance_linear_offset,
            self.max_compliance_linear_offset,
        )

    def _rotation_acceleration_body(
        self,
        rotation_offset_body: np.ndarray,
        angular_velocity_body: np.ndarray,
        torque_body: np.ndarray,
    ) -> np.ndarray:
        mass = self.admittance_mass[3:]
        damping = self.admittance_damping[3:]
        stiffness = self.admittance_stiffness[3:]
        phi_body = self._log_so3(rotation_offset_body)
        return (
            torque_body - damping * angular_velocity_body - stiffness * phi_body
        ) / mass

    def _integrate_rotation_admittance_rk4(
        self,
        *,
        dt: float,
        torque_body: np.ndarray,
    ) -> None:
        """用 body/right-invariant SO(3) RK4 积分旋转导纳。"""
        h = float(dt)
        rotation_offset = self._exp_so3(self.compliance_rotation_body)
        omega = self.compliance_angular_velocity_body

        k1_w = self._rotation_acceleration_body(rotation_offset, omega, torque_body)

        omega2 = omega + 0.5 * h * k1_w
        rotation2 = rotation_offset @ self._exp_so3(omega * 0.5 * h)
        k2_w = self._rotation_acceleration_body(rotation2, omega2, torque_body)

        omega3 = omega + 0.5 * h * k2_w
        rotation3 = rotation_offset @ self._exp_so3(omega2 * 0.5 * h)
        k3_w = self._rotation_acceleration_body(rotation3, omega3, torque_body)

        omega4 = omega + h * k3_w
        rotation4 = rotation_offset @ self._exp_so3(omega3 * h)
        k4_w = self._rotation_acceleration_body(rotation4, omega4, torque_body)

        self.compliance_angular_velocity_body = omega + h / 6.0 * (
            k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w
        )
        self.compliance_angular_velocity_body = np.clip(
            self.compliance_angular_velocity_body,
            -self.max_compliance_angular_velocity,
            self.max_compliance_angular_velocity,
        )
        rotation_next = rotation_offset @ self._exp_so3(
            self.compliance_angular_velocity_body * h
        )
        phi_next = self._log_so3(self._project_rotation_matrix(rotation_next))
        phi_norm = np.linalg.norm(phi_next)
        if phi_norm > self.max_compliance_angular_offset > 0.0:
            phi_next *= self.max_compliance_angular_offset / phi_norm
        self.compliance_rotation_body = phi_next

    @staticmethod
    def _exp_so3(rotvec: np.ndarray) -> np.ndarray:
        return exp_so3(rotvec)

    @staticmethod
    def _log_so3(rotation_matrix: np.ndarray) -> np.ndarray:
        return log_so3(rotation_matrix)

    @staticmethod
    def _project_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
        return project_rotation_matrix(rotation_matrix)

    def _compute_local_pose_error(
        self,
        *,
        current_transform: np.ndarray,
        target_transform: np.ndarray,
    ) -> np.ndarray:
        current_inverse = np.eye(4, dtype=np.float64)
        current_rotation = current_transform[:3, :3]
        current_position = current_transform[:3, 3]
        current_inverse[:3, :3] = current_rotation.T
        current_inverse[:3, 3] = -(current_rotation.T @ current_position)
        relative_transform = current_inverse @ target_transform
        return log_se3(relative_transform)

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
