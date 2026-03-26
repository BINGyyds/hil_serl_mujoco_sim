# robot_motion.py

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class RobotMotion:
    """机器人运动规划类。"""

    def __init__(self) -> None:
        pass

    def motion_plan_joint(self, t: float, tf: float, q0: np.ndarray, qf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用五次多项式规划一条平滑关节轨迹。"""
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
