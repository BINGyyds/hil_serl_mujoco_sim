"""Internal SE(3) geometric impedance math helpers.

This module intentionally mirrors only the subset of the repository-root
``SE3_impedance`` package that the insertion mainline needs at runtime.
"""

from __future__ import annotations

import numpy as np

from serl_robot_infra.common.se3_utils import dexp_inv_so3, dexp_so3, log_so3, skew


class SE3GeometryImpedance:
    """Lightweight geometric control utilities on :math:`SE(3)`."""

    @staticmethod
    def near_zero(value: float, *, atol: float = 1e-7) -> bool:
        return abs(float(value)) < atol

    @staticmethod
    def _bracket(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        skew_left = skew(left)
        skew_right = skew(right)
        return skew_left @ skew_right + skew_right @ skew_left

    def log_map_se3(self, transform: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        rotvec = log_so3(rotation)
        if self.near_zero(np.linalg.norm(rotvec)):
            linear = translation.copy()
        else:
            linear = dexp_inv_so3(rotvec) @ translation
        return np.concatenate([linear, rotvec])

    def dexp_se3(self, lamda: np.ndarray) -> np.ndarray:
        lamda = np.asarray(lamda, dtype=np.float64).reshape((6,))
        eta = lamda[:3]
        xi = lamda[3:]
        dexp_xi = dexp_so3(xi)
        c_mat = self._c_xi(xi, eta)
        zeros = np.zeros((3, 3), dtype=np.float64)
        return np.block([[dexp_xi, c_mat], [zeros, dexp_xi]])

    def dexp_inv_se3(self, lamda: np.ndarray) -> np.ndarray:
        lamda = np.asarray(lamda, dtype=np.float64).reshape((6,))
        eta = lamda[:3]
        xi = lamda[3:]
        dexp_inv_xi = dexp_inv_so3(xi)
        d_mat = self._d_xi(xi, eta)
        zeros = np.zeros((3, 3), dtype=np.float64)
        return np.block([[dexp_inv_xi, d_mat], [zeros, dexp_inv_xi]])

    def dexp_dot_se3(self, lamda: np.ndarray, lamda_dot: np.ndarray) -> np.ndarray:
        lamda = np.asarray(lamda, dtype=np.float64).reshape((6,))
        lamda_dot = np.asarray(lamda_dot, dtype=np.float64).reshape((6,))
        eta, xi = lamda[:3], lamda[3:]
        eta_dot, xi_dot = lamda_dot[:3], lamda_dot[3:]
        c_xi_dot = self._c_xi(xi, xi_dot)
        c_dot = self._c_dot_xi(xi, xi_dot, eta, eta_dot)
        zeros = np.zeros((3, 3), dtype=np.float64)
        return np.block([[c_xi_dot, c_dot], [zeros, c_xi_dot]])

    def Ad_T(self, transform: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        zeros = np.zeros((3, 3), dtype=np.float64)
        return np.block(
            [
                [rotation, skew(translation) @ rotation],
                [zeros, rotation],
            ]
        )

    def ad_V(self, twist: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=np.float64).reshape((6,))
        linear = twist[:3]
        angular = twist[3:]
        zeros = np.zeros((3, 3), dtype=np.float64)
        return np.block(
            [
                [skew(angular), skew(linear)],
                [zeros, skew(angular)],
            ]
        )

    def _c_xi(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64).reshape((3,))
        eta = np.asarray(eta, dtype=np.float64).reshape((3,))
        norm_xi = float(np.linalg.norm(xi))
        skew_eta = skew(eta)
        if self.near_zero(norm_xi):
            return 0.5 * skew_eta

        skew_xi = skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        skew_eta_xi = self._bracket(eta, xi)
        s = 2.0 * np.sin(0.5 * norm_xi) / norm_xi
        c = np.cos(0.5 * norm_xi)
        alpha = s * c
        beta = s * s
        xi_dot_eta = float(np.dot(xi, eta))

        term1 = 0.5 * beta * skew_eta
        term2 = ((1.0 - alpha) / (norm_xi**2)) * skew_eta_xi
        term3 = ((alpha - beta) / (norm_xi**2)) * xi_dot_eta * skew_xi
        term4 = -(1.0 / (norm_xi**2)) * (
            (3.0 * (1.0 - alpha) / (norm_xi**2)) - 0.5 * beta
        ) * xi_dot_eta * skew_xi_sq
        return term1 + term2 + term3 + term4

    def _d_xi(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64).reshape((3,))
        eta = np.asarray(eta, dtype=np.float64).reshape((3,))
        norm_xi = float(np.linalg.norm(xi))
        skew_eta = skew(eta)
        if self.near_zero(norm_xi):
            return -0.5 * skew_eta

        skew_xi = skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        skew_eta_xi = self._bracket(eta, xi)
        beta = 2.0 * (1.0 - np.cos(norm_xi)) / (norm_xi**2)
        gamma = (norm_xi / 2.0) / np.tan(norm_xi / 2.0)
        factor_gamma = (1.0 / norm_xi - 0.5 / np.tan(norm_xi / 2.0)) / norm_xi
        xi_dot_eta = float(np.dot(xi, eta))

        term1 = -0.5 * skew_eta
        term2 = factor_gamma * skew_eta_xi
        term3 = (1.0 / (norm_xi**2)) * (
            ((1.0 / beta) + gamma - 2.0) / (norm_xi**2)
        ) * xi_dot_eta * skew_xi_sq
        return term1 + term2 + term3

    def _c_dot_xi(
        self,
        xi: np.ndarray,
        xi_dot: np.ndarray,
        eta: np.ndarray,
        eta_dot: np.ndarray,
    ) -> np.ndarray:
        norm_xi = float(np.linalg.norm(xi))
        if self.near_zero(norm_xi):
            return 0.5 * skew(eta_dot) + (1.0 / 6.0) * self._bracket(xi_dot, eta)

        skew_xi = skew(xi)
        skew_xi_sq = skew_xi @ skew_xi
        xi_dot_xi = float(np.dot(xi, xi_dot))

        s = 2.0 * np.sin(0.5 * norm_xi) / norm_xi
        c = np.cos(0.5 * norm_xi)
        alpha = s * c
        beta = s * s

        gamma_1 = (1.0 - alpha) / (norm_xi**2)
        gamma_2 = (alpha - beta) / (norm_xi**2)
        gamma_3 = (0.5 * beta - 3.0 * gamma_1) / (norm_xi**2)

        zeta_eta = float(np.dot(xi, eta) * xi_dot_xi / (norm_xi**2))
        delta_0 = float(np.dot(xi_dot, eta) + np.dot(xi, eta_dot))
        delta_1 = (
            np.dot(xi, eta) * skew(xi_dot)
            + np.dot(xi, xi_dot) * skew(eta)
            + (delta_0 - 4.0 * zeta_eta) * skew_xi
        )
        delta_2 = (
            np.dot(xi, eta) * self._bracket(xi, xi_dot)
            + np.dot(xi, xi_dot) * self._bracket(xi, eta)
            + (delta_0 - 5.0 * zeta_eta) * skew_xi_sq
        )

        term1 = 0.5 * beta * (skew(eta_dot) - zeta_eta * skew_xi)
        term2 = gamma_1 * (
            self._bracket(eta_dot, xi) + self._bracket(eta, xi_dot) + zeta_eta * skew_xi
        )
        term3 = gamma_2 * (delta_1 + zeta_eta * skew_xi_sq)
        term4 = gamma_3 * delta_2
        return term1 + term2 + term3 + term4
