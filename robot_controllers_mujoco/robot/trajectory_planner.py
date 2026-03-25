import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryPlanner:
    """从 CSV 读取轨迹，缺省时回退到一条 Panda 安全测试轨迹。"""

    def __init__(self) -> None:
        pass

    def plan_trajectory(
        self,
        t: float,
        tf: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用五次多项式规划一条平滑关节轨迹。"""
        # q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # qf = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5])
        q0 = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        qf = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        normalized_t = t / tf

        qt = np.zeros(7)
        qt_dot = np.zeros(7)
        qt_ddot = np.zeros(7)

        for i in range(len(q0)):
            delta = qf[i] - q0[i]
            qt[i] = q0[i] + delta * (10 * normalized_t**3 - 15 * normalized_t**4 + 6 * normalized_t**5)
            qt_dot[i] = delta * (30 * normalized_t**2 - 60 * normalized_t**3 + 30 * normalized_t**4) / tf
            qt_ddot[i] = delta * (60 * normalized_t - 180 * normalized_t**2 + 120 * normalized_t**3) / (tf**2)

        return qt, qt_dot, qt_ddot

    # def plan_trajectory(self, t) -> np.ndarray:
    #     """自行规划轨迹（用于测试）"""
    #     tf = 50.0
    #     q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #     qf = [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5]
    #     qt = np.zeros(7)
    #     for i in range(7):
    #         qt[i] = q0[i] + (qf[i] - q0[i]) * (10 * pow(t/tf, 3) - 15 * pow(t/tf, 4) + 6 * pow(t/tf, 5))
    #     return qt
