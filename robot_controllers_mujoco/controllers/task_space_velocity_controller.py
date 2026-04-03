"""
Shared task-space velocity controller used by both teleop and MuJoCo envs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class TaskSpaceVelocityController:
    """Stateful task-space controller with target-pose integration and DLS IK."""

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
    ) -> None:
        self.input_deadzone = float(input_deadzone)
        self.input_filter_alpha = float(np.clip(input_filter_alpha, 0.0, 1.0))
        self.max_reference_linear_velocity = float(max_reference_linear_velocity)
        self.max_reference_angular_velocity = float(max_reference_angular_velocity)
        self.pose_error_gain = float(pose_error_gain)
        self.ik_damping = float(ik_damping)
        self.nullspace_gain = float(nullspace_gain)
        self.joint_velocity_limits = np.asarray(
            joint_velocity_limits, dtype=np.float64
        ).copy()
        self.joint_limit_margin = float(joint_limit_margin)
        self.lower_position_limits = np.asarray(
            lower_position_limits, dtype=np.float64
        ).copy()
        self.upper_position_limits = np.asarray(
            upper_position_limits, dtype=np.float64
        ).copy()

        self.filtered_command = np.zeros(6, dtype=np.float64)
        self.target_transform: Optional[np.ndarray] = None
        self.posture_reference: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        current_transform: np.ndarray,
        current_q: np.ndarray,
        posture_reference: Optional[np.ndarray] = None,
    ) -> None:
        """Reset controller state to the current robot pose."""
        self.target_transform = np.asarray(current_transform, dtype=np.float64).copy()
        if posture_reference is None:
            self.posture_reference = np.asarray(current_q, dtype=np.float64).copy()
        else:
            self.posture_reference = np.asarray(
                posture_reference, dtype=np.float64
            ).copy()
        self.filtered_command[:] = 0.0

    def get_target_transform(self) -> Optional[np.ndarray]:
        """Return a copy of the currently integrated target pose."""
        if self.target_transform is None:
            return None
        return self.target_transform.copy()

    def compute_command(
        self,
        *,
        raw_action: np.ndarray,
        current_transform: np.ndarray,
        jacobian: np.ndarray,
        current_q: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute joint velocity command from a normalized 6-D input."""
        if self.target_transform is None or self.posture_reference is None:
            self.reset(
                current_transform=current_transform,
                current_q=current_q,
            )

        filtered_command = self._filter_input(np.asarray(raw_action, dtype=np.float64))
        self._integrate_target_transform(filtered_command, dt=float(dt))
        pose_error = self._compute_local_pose_error(
            current_transform=np.asarray(current_transform, dtype=np.float64),
            target_transform=self.target_transform,
        )
        return self._compute_joint_velocity_command(
            current_q=np.asarray(current_q, dtype=np.float64),
            jacobian=np.asarray(jacobian, dtype=np.float64),
            pose_error=pose_error,
        )

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

    def _integrate_target_transform(self, filtered_command: np.ndarray, dt: float) -> None:
        if self.target_transform is None:
            raise RuntimeError("Task-space target transform is not initialized.")

        delta_translation_local = (
            filtered_command[:3] * dt * self.max_reference_linear_velocity
        )
        delta_rotation_local = (
            filtered_command[3:] * dt * self.max_reference_angular_velocity
        )

        target_transform = self.target_transform.copy()
        target_rotation = target_transform[:3, :3]
        target_position = target_transform[:3, 3]

        target_position = target_position + target_rotation @ delta_translation_local
        target_rotation = target_rotation @ Rotation.from_rotvec(
            delta_rotation_local
        ).as_matrix()

        self.target_transform[:3, :3] = target_rotation
        self.target_transform[:3, 3] = target_position

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

    def _apply_joint_limit_guard(
        self,
        current_q: np.ndarray,
        qdot_command: np.ndarray,
    ) -> np.ndarray:
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
