"""Lightweight SE(3) helpers internal to :mod:`serl_robot_infra`.

The runtime code intentionally depends only on this module instead of importing
the experimental ``SE3_impedance`` stack from the repository root.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def near_zero(value: float, *, atol: float = 1e-7) -> bool:
    return abs(float(value)) < atol


def skew(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape((3,))
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=np.float64,
    )


def project_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def exp_so3(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape((3,))
    return Rotation.from_rotvec(rotvec).as_matrix()


def log_so3(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = project_rotation_matrix(rotation_matrix)
    return Rotation.from_matrix(rotation_matrix).as_rotvec()


def dexp_so3(rotvec: np.ndarray) -> np.ndarray:
    """Right/left Jacobian variant used by the Kim/Seo SE(3) formulation.

    With the [v, w] convention used in this repo, this matches the matrix used
    in ``SE3_impedance.geometry_impedance_control.exponential_map_se3``.
    """

    rotvec = np.asarray(rotvec, dtype=np.float64).reshape((3,))
    theta = np.linalg.norm(rotvec)
    identity = np.eye(3, dtype=np.float64)
    if near_zero(theta):
        return identity

    s = 2.0 * np.sin(0.5 * theta) / theta
    c = np.cos(0.5 * theta)
    alpha = s * c
    beta = s * s
    skew_rotvec = skew(rotvec)
    skew_rotvec_sq = skew_rotvec @ skew_rotvec
    return (
        identity
        + 0.5 * beta * skew_rotvec
        + ((1.0 - alpha) / (theta**2)) * skew_rotvec_sq
    )


def dexp_inv_so3(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape((3,))
    theta = np.linalg.norm(rotvec)
    identity = np.eye(3, dtype=np.float64)
    if near_zero(theta):
        return identity

    skew_rotvec = skew(rotvec)
    skew_rotvec_sq = skew_rotvec @ skew_rotvec
    factor = (1.0 / theta - 0.5 / np.tan(theta / 2.0)) / theta
    return identity - 0.5 * skew_rotvec + factor * skew_rotvec_sq


def exp_se3(body_twist: np.ndarray) -> np.ndarray:
    body_twist = np.asarray(body_twist, dtype=np.float64).reshape((6,))
    translation = body_twist[:3]
    rotation = body_twist[3:]

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = exp_so3(rotation)
    transform[:3, 3] = dexp_so3(rotation) @ translation
    return transform


def log_se3(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
    rotation = project_rotation_matrix(transform[:3, :3])
    translation = transform[:3, 3]
    rotvec = log_so3(rotation)
    if near_zero(np.linalg.norm(rotvec)):
        linear = translation.copy()
    else:
        linear = dexp_inv_so3(rotvec) @ translation
    return np.concatenate([linear, rotvec])


def make_transform(*, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = project_rotation_matrix(rotation)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape((3,))
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape((4, 4))
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transforms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, dtype=np.float64).reshape((4, 4)) @ np.asarray(
        right, dtype=np.float64
    ).reshape((4, 4))


def transform_translation_in_frame(
    *,
    point_world: np.ndarray,
    frame_transform: np.ndarray,
) -> np.ndarray:
    frame_rotation = np.asarray(frame_transform, dtype=np.float64)[:3, :3]
    frame_translation = np.asarray(frame_transform, dtype=np.float64)[:3, 3]
    return frame_rotation.T @ (
        np.asarray(point_world, dtype=np.float64).reshape((3,)) - frame_translation
    )


def rotation_matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(project_rotation_matrix(rotation_matrix)).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
