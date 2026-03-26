# robot_model.py

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


class RobotModel:
    """使用 Pinocchio 基于 MJCF 构建 Panda 的运动学和动力学模型。"""

    def __init__(self, mjcf_path: Path, end_effector_frame_name: str) -> None:
        self.mjcf_path = Path(mjcf_path)
        try:
            self.model = cast(pin.Model, pin.buildModelFromMJCF(str(self.mjcf_path)))
            self.data = cast(pin.Data, self.model.createData())

            if not self.model.existFrame(end_effector_frame_name):
                raise ValueError(f"MJCF 中未找到末端坐标系: {end_effector_frame_name}")

            self.end_effector_frame_id = int(
                self.model.getFrameId(end_effector_frame_name)
            )
            logger.info("Pinocchio 动力学模型构建成功: %s", self.mjcf_path)
        except Exception:
            logger.exception("从 MJCF 构建 Pinocchio 模型失败。")
            raise

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行正向运动学计算并返回末端位姿。"""
        q_array = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q_array)
        pin.updateFramePlacements(self.model, self.data)

        frame_placements = cast(Any, self.data.oMf)
        pose = cast(pin.SE3, frame_placements[self.end_effector_frame_id])
        position = np.asarray(pose.translation, dtype=np.float64).copy()
        rotation = np.asarray(pose.rotation, dtype=np.float64).copy()
        return position, rotation

    def inverse_kinematics(
        self,
        target_transform: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        tolerance: float = 1e-4,
        max_iterations: int = 200,
        damping: float = 1e-6,
        step_scale: float = 0.7,
        max_step_norm: float = 0.2,
    ) -> np.ndarray:
        """根据末端目标位姿数值求解关节角度。

        Args:
            target_transform: 目标末端位姿的 4x4 齐次变换矩阵。
            q_init: 逆运动学初值，形状为 (nq,)；为 None 时使用关节限位中点初始化。
            tolerance: 6D 位姿误差的收敛阈值。
            max_iterations: 最大迭代次数。
            damping: 阻尼最小二乘中的阻尼系数。
            step_scale: 每次关节更新的缩放系数。
            max_step_norm: 单次关节更新向量的最大 2 范数。

        Returns:
            一组满足目标末端位姿的关节角，形状为 (nq,)。

        Notes:
            使用 Pinocchio 的 SE(3) 对数映射构造 6D 位姿误差，并配合 Jlog6 修正后的
            frame Jacobian 做阻尼最小二乘迭代。
        """
        target_matrix = np.asarray(target_transform, dtype=np.float64).reshape((4, 4))
        target_pose = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
        nq = int(cast(Any, self.model.nq))

        if q_init is None:
            lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
            upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
            q = np.zeros((nq,), dtype=np.float64)

            finite_mask = np.isfinite(lower) & np.isfinite(upper)
            q[finite_mask] = 0.5 * (lower[finite_mask] + upper[finite_mask])
        else:
            q = np.asarray(q_init, dtype=np.float64).reshape((nq,)).copy()

        lower = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        finite_mask = np.isfinite(lower) & np.isfinite(upper)

        for _ in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            frame_placements = cast(Any, self.data.oMf)
            current_pose = cast(pin.SE3, frame_placements[self.end_effector_frame_id])
            pose_error = current_pose.actInv(target_pose)
            error = np.asarray(pin.log(pose_error).vector, dtype=np.float64).reshape(6)

            if np.linalg.norm(error) < tolerance:
                return q.copy()

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.end_effector_frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jacobian = -np.asarray(
                pin.Jlog6(pose_error.inverse()) @ jacobian,
                dtype=np.float64,
            )

            jjt = jacobian @ jacobian.T
            delta_q = -jacobian.T @ np.linalg.solve(
                jjt + damping * np.eye(6, dtype=np.float64),
                error,
            )

            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > max_step_norm:
                delta_q *= max_step_norm / delta_norm

            q = pin.integrate(self.model, q, step_scale * delta_q)
            q = np.asarray(q, dtype=np.float64)
            q[finite_mask] = np.clip(q[finite_mask], lower[finite_mask], upper[finite_mask])

        raise ValueError(
            f"逆运动学未收敛: iterations={max_iterations}, "
            f"target_position={target_matrix[:3, 3].tolist()}"
        )

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """执行逆动力学计算（RNEA）。"""
        q_array = np.asarray(q, dtype=np.float64)
        v_array = np.asarray(v, dtype=np.float64)
        a_array = np.asarray(a, dtype=np.float64)
        tau = pin.rnea(self.model, self.data, q_array, v_array, a_array)
        return np.asarray(tau, dtype=np.float64).copy()
