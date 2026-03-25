# robot_model.py

import logging
from pathlib import Path
from typing import Any, Tuple, cast

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

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """执行逆动力学计算（RNEA）。"""
        q_array = np.asarray(q, dtype=np.float64)
        v_array = np.asarray(v, dtype=np.float64)
        a_array = np.asarray(a, dtype=np.float64)
        tau = pin.rnea(self.model, self.data, q_array, v_array, a_array)
        return np.asarray(tau, dtype=np.float64).copy()
