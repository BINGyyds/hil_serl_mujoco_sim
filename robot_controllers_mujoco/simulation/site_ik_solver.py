"""共享的 MuJoCo site 位姿逆运动学求解器。

该模块刻意不依赖 Gym 环境或 debug 控制器，使 reset 阶段 IK 和 planned
模式 IK 可以复用同一套带关节边界约束的非线性最小二乘后端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mujoco
import numpy as np
from mujoco import minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SiteIKOptions:
    """有界 site 位姿 IK 的调参项。"""

    max_iterations: int = 80
    tolerance: float = 1e-4
    position_weight: float = 1.0
    rotation_weight: float = 0.05
    seed_weight: float = 0.01
    center_weight: float = 0.002
    position_tolerance: float | None = None
    rotation_tolerance: float | None = None
    xtol: float = 1e-8
    gtol: float = 1e-8


@dataclass(frozen=True)
class SiteIKResult:
    """``solve_site_pose_ik`` 返回的求解结果。"""

    q: np.ndarray
    position_error_norm: float
    rotation_error_norm: float
    objective: float
    seed_index: int
    converged: bool


def solve_site_pose_ik(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_transform: np.ndarray,
    q_seed: np.ndarray,
    qpos_addrs: np.ndarray,
    dof_addrs: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    site_id: int,
    options: SiteIKOptions,
    extra_seeds: Sequence[np.ndarray] | None = None,
) -> SiteIKResult:
    """为单个 MuJoCo site 求解带关节边界约束的 IK。

    优化器只修改传入的机械臂关节子集。其他广义坐标保持为调用方传入
    scratch ``data.qpos`` 中的原值。
    """
    target_matrix = np.asarray(target_transform, dtype=np.float64).reshape((4, 4))
    target_position = target_matrix[:3, 3].copy()
    target_rotation = target_matrix[:3, :3].copy()
    q_seed = np.asarray(q_seed, dtype=np.float64).reshape((-1,))
    qpos_addrs = np.asarray(qpos_addrs, dtype=np.int32).reshape((-1,))
    dof_addrs = np.asarray(dof_addrs, dtype=np.int32).reshape((-1,))
    q_low = np.asarray(q_low, dtype=np.float64).reshape(q_seed.shape)
    q_high = np.asarray(q_high, dtype=np.float64).reshape(q_seed.shape)

    if q_seed.shape != qpos_addrs.shape or q_seed.shape != dof_addrs.shape:
        raise ValueError(
            "q_seed、qpos_addrs 和 dof_addrs 必须描述同一组关节。"
        )

    seeds = _deduplicate_seeds(
        [q_seed, *(extra_seeds or ())],
        q_low=q_low,
        q_high=q_high,
    )
    if not seeds:
        raise ValueError("至少需要一个有限的 IK 初值。")

    base_qpos = data.qpos.copy()
    base_qvel = data.qvel.copy()
    finite_limits = np.isfinite(q_low) & np.isfinite(q_high)
    lower_bounds = np.where(finite_limits, q_low, -2.0 * np.pi)
    upper_bounds = np.where(finite_limits, q_high, 2.0 * np.pi)
    joint_center = 0.5 * (lower_bounds + upper_bounds)
    joint_scale = np.maximum(upper_bounds - lower_bounds, 1.0)

    best: SiteIKResult | None = None
    for seed_index, seed in enumerate(seeds):
        residual_fn = _make_residual(
            model=model,
            data=data,
            base_qpos=base_qpos,
            base_qvel=base_qvel,
            qpos_addrs=qpos_addrs,
            target_position=target_position,
            target_rotation=target_rotation,
            site_id=site_id,
            seed=seed,
            joint_center=joint_center,
            joint_scale=joint_scale,
            options=options,
        )
        q_solution, trace = minimize.least_squares(
            seed,
            residual_fn,
            bounds=[lower_bounds, upper_bounds],
            max_iter=int(options.max_iterations),
            xtol=float(options.xtol),
            gtol=float(options.gtol),
            verbose=minimize.Verbosity.SILENT,
        )
        q_solution = np.asarray(q_solution, dtype=np.float64).reshape(seed.shape)
        result = _evaluate_solution(
            model=model,
            data=data,
            base_qpos=base_qpos,
            base_qvel=base_qvel,
            qpos_addrs=qpos_addrs,
            target_position=target_position,
            target_rotation=target_rotation,
            site_id=site_id,
            q=q_solution,
            seed_index=seed_index,
            trace=trace,
            options=options,
        )
        if best is None or _ranking_key(result) < _ranking_key(best):
            best = result

    if best is None:
        raise RuntimeError("IK 求解器没有评估任何候选初值。")

    if not best.converged:
        raise ValueError(
            "MuJoCo site IK 未收敛: "
            f"seed_index={best.seed_index}, "
            f"position_error={best.position_error_norm:.6g}, "
            f"rotation_error={best.rotation_error_norm:.6g}, "
            f"objective={best.objective:.6g}, "
            f"target_position={target_position.tolist()}"
        )
    return best


def _deduplicate_seeds(
    seeds: Iterable[np.ndarray],
    *,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for seed in seeds:
        candidate = np.asarray(seed, dtype=np.float64).reshape(q_low.shape)
        if not np.all(np.isfinite(candidate)):
            continue
        candidate = np.clip(candidate, q_low, q_high)
        if any(np.linalg.norm(candidate - existing) < 1e-9 for existing in unique):
            continue
        unique.append(candidate)
    return unique


def _make_residual(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_qpos: np.ndarray,
    base_qvel: np.ndarray,
    qpos_addrs: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    site_id: int,
    seed: np.ndarray,
    joint_center: np.ndarray,
    joint_scale: np.ndarray,
    options: SiteIKOptions,
):
    def residual(q_batch: np.ndarray) -> np.ndarray:
        q_candidates = np.asarray(q_batch, dtype=np.float64)
        if q_candidates.ndim == 1:
            q_candidates = q_candidates.reshape((-1, 1))

        residuals = np.zeros((6 + 2 * seed.size, q_candidates.shape[1]), dtype=np.float64)
        for column in range(q_candidates.shape[1]):
            q = q_candidates[:, column]
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qpos[qpos_addrs] = q
            mujoco.mj_forward(model, data)

            site_position = data.site_xpos[site_id]
            site_rotation = data.site_xmat[site_id].reshape(3, 3)
            position_error = target_position - site_position
            rotation_error = Rotation.from_matrix(
                target_rotation @ site_rotation.T
            ).as_rotvec()

            residuals[:3, column] = options.position_weight * position_error
            residuals[3:6, column] = options.rotation_weight * rotation_error
            residuals[6 : 6 + seed.size, column] = (
                options.seed_weight * (q - seed) / joint_scale
            )
            residuals[6 + seed.size :, column] = (
                options.center_weight * (q - joint_center) / joint_scale
            )
        return residuals

    return residual


def _evaluate_solution(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_qpos: np.ndarray,
    base_qvel: np.ndarray,
    qpos_addrs: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    site_id: int,
    q: np.ndarray,
    seed_index: int,
    trace: Sequence,
    options: SiteIKOptions,
) -> SiteIKResult:
    data.qpos[:] = base_qpos
    data.qvel[:] = base_qvel
    data.qpos[qpos_addrs] = q
    mujoco.mj_forward(model, data)

    site_position = data.site_xpos[site_id].copy()
    site_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    position_error = target_position - site_position
    rotation_error = Rotation.from_matrix(target_rotation @ site_rotation.T).as_rotvec()
    position_error_norm = float(np.linalg.norm(position_error))
    rotation_error_norm = float(np.linalg.norm(rotation_error))
    position_tolerance = (
        options.tolerance
        if options.position_tolerance is None
        else options.position_tolerance
    )
    rotation_tolerance = (
        options.tolerance
        if options.rotation_tolerance is None
        else options.rotation_tolerance
    )
    objective = float(trace[-1].objective) if trace else np.inf
    converged = (
        position_error_norm <= position_tolerance
        and rotation_error_norm <= rotation_tolerance
    )
    return SiteIKResult(
        q=q.copy(),
        position_error_norm=position_error_norm,
        rotation_error_norm=rotation_error_norm,
        objective=objective,
        seed_index=seed_index,
        converged=converged,
    )


def _ranking_key(result: SiteIKResult) -> tuple[float, float, float]:
    return (
        result.position_error_norm,
        result.rotation_error_norm,
        result.objective,
    )
