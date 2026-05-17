"""Torque-level SE(3) geometric impedance controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from serl_robot_infra.common.se3_gic import SE3GeometryImpedance
from serl_robot_infra.common.se3_utils import invert_transform


@dataclass(frozen=True)
class SE3TorqueControlResult:
    joint_torque: np.ndarray
    pose_error_body: np.ndarray
    twist_error_body: np.ndarray
    reference_twist_dot_body: np.ndarray


class SE3TorqueController:
    """SE(3) geometric impedance controller at the joint-torque level."""

    def __init__(
        self,
        *,
        gic: Optional[SE3GeometryImpedance] = None,
        apparent_inertia: Optional[np.ndarray] = None,
        lambda_stiffness: Optional[np.ndarray] = None,
        physical_damping: Optional[np.ndarray] = None,
        q_nominal: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
    ) -> None:
        self.gic = gic or SE3GeometryImpedance()
        self.A = self._coerce_diag(
            apparent_inertia,
            default=np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1], dtype=np.float64),
        )
        self.K_lam = self._coerce_diag(
            lambda_stiffness,
            default=np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5], dtype=np.float64),
        )
        if physical_damping is None:
            self.D = 2.0 * np.sqrt(self.A @ self.K_lam)
        else:
            self.D = self._coerce_diag(physical_damping, default=None)
        self.q_nominal = None if q_nominal is None else np.asarray(q_nominal, dtype=np.float64).copy()
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)

    @staticmethod
    def _coerce_diag(value: Optional[np.ndarray], *, default: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            if default is None:
                raise ValueError("A default diagonal matrix is required.")
            vector = np.asarray(default, dtype=np.float64).reshape((-1,))
            return np.diag(vector)
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (6,):
            return np.diag(array)
        if array.shape == (6, 6):
            return array.copy()
        raise ValueError(f"Expected shape (6,) or (6, 6), got {array.shape}")

    def compute_torque(
        self,
        *,
        q: np.ndarray,
        q_dot: np.ndarray,
        current_transform: np.ndarray,
        current_twist_body: np.ndarray,
        jacobian_body: np.ndarray,
        mass_matrix: np.ndarray,
        bias_forces: np.ndarray,
        desired_transform: np.ndarray,
        desired_twist_body: Optional[np.ndarray] = None,
        desired_twist_dot_body: Optional[np.ndarray] = None,
        external_wrench_body: Optional[np.ndarray] = None,
    ) -> SE3TorqueControlResult:
        q = np.asarray(q, dtype=np.float64).reshape((-1,))
        q_dot = np.asarray(q_dot, dtype=np.float64).reshape((-1,))
        current_transform = np.asarray(current_transform, dtype=np.float64).reshape((4, 4))
        current_twist_body = np.asarray(current_twist_body, dtype=np.float64).reshape((6,))
        jacobian_body = np.asarray(jacobian_body, dtype=np.float64).reshape((6, -1))
        mass_matrix = np.asarray(mass_matrix, dtype=np.float64).reshape((q.size, q.size))
        bias_forces = np.asarray(bias_forces, dtype=np.float64).reshape((q.size,))
        desired_transform = np.asarray(desired_transform, dtype=np.float64).reshape((4, 4))
        desired_twist_body = (
            np.zeros(6, dtype=np.float64)
            if desired_twist_body is None
            else np.asarray(desired_twist_body, dtype=np.float64).reshape((6,))
        )
        desired_twist_dot_body = (
            np.zeros(6, dtype=np.float64)
            if desired_twist_dot_body is None
            else np.asarray(desired_twist_dot_body, dtype=np.float64).reshape((6,))
        )
        external_wrench_body = (
            np.zeros(6, dtype=np.float64)
            if external_wrench_body is None
            else np.asarray(external_wrench_body, dtype=np.float64).reshape((6,))
        )

        relative_transform = invert_transform(current_transform) @ desired_transform
        pose_error_body = self.gic.log_map_se3(relative_transform)
        adjoint_relative = self.gic.Ad_T(relative_transform)
        twist_error_body = adjoint_relative @ desired_twist_body - current_twist_body
        pose_error_dot_body = self.gic.dexp_inv_se3(pose_error_body) @ twist_error_body

        dexp_pose = self.gic.dexp_se3(pose_error_body)
        dexp_pose_dot = self.gic.dexp_dot_se3(pose_error_body, pose_error_dot_body)
        A_lam = dexp_pose.T @ self.A @ dexp_pose
        D_lam = (dexp_pose.T @ self.D @ dexp_pose) + (dexp_pose.T @ self.A @ dexp_pose_dot)
        gamma = dexp_pose.T @ (-external_wrench_body)

        lambda_ddot_ref = np.linalg.solve(
            A_lam,
            gamma - D_lam @ pose_error_dot_body - self.K_lam @ pose_error_body,
        )
        twist_error_dot_body = dexp_pose @ lambda_ddot_ref + dexp_pose_dot @ pose_error_dot_body
        reference_twist_dot_body = (
            adjoint_relative @ desired_twist_dot_body
            - twist_error_dot_body
            + self.gic.ad_V(twist_error_body) @ current_twist_body
        )

        mass_matrix_inv = np.linalg.inv(mass_matrix)
        lambda_inv = jacobian_body @ mass_matrix_inv @ jacobian_body.T
        operational_inertia = np.linalg.inv(lambda_inv + 1e-5 * np.eye(6, dtype=np.float64))
        dynamic_consistent_pinv = mass_matrix_inv @ jacobian_body.T @ operational_inertia

        tau_task = (
            jacobian_body.T @ operational_inertia @ reference_twist_dot_body
            + bias_forces
            - jacobian_body.T @ external_wrench_body
        )

        tau_null = np.zeros_like(q, dtype=np.float64)
        if self.q_nominal is not None:
            nullspace_projector = np.eye(q.size, dtype=np.float64) - jacobian_body.T @ dynamic_consistent_pinv.T
            tau_0 = (
                -self.nullspace_stiffness * (q - self.q_nominal)
                - self.nullspace_damping * q_dot
            )
            tau_null = nullspace_projector @ tau_0

        return SE3TorqueControlResult(
            joint_torque=tau_task + tau_null,
            pose_error_body=pose_error_body,
            twist_error_body=twist_error_body,
            reference_twist_dot_body=reference_twist_dot_body,
        )
