# robot_motion.py

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

if TYPE_CHECKING:
    from robot_controllers_mujoco.simulation.robot_model import RobotModel

IKSolver = Callable[[np.ndarray, np.ndarray], np.ndarray]

logger = logging.getLogger(__name__)


class RobotMotion:
    """机器人运动规划类。"""

    def __init__(self) -> None:
        self.trajectory_duration: float = 0.0
        self.joint_position_splines: List[BSpline] = []
        self.joint_velocity_splines: List[BSpline] = []
        self.joint_acceleration_splines: List[BSpline] = []

    def motion_plan_joint(
        self, t: float, tf: float, q0: np.ndarray, qf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用五次多项式规划一条平滑关节轨迹。"""
        normalized_t = np.clip(t / tf, 0.0, 1.0)

        qt = np.zeros(7)
        qt_dot = np.zeros(7)
        qt_ddot = np.zeros(7)

        for i in range(len(q0)):
            delta = qf[i] - q0[i]
            qt[i] = q0[i] + delta * (
                10 * normalized_t**3 - 15 * normalized_t**4 + 6 * normalized_t**5
            )
            qt_dot[i] = (
                delta
                * (30 * normalized_t**2 - 60 * normalized_t**3 + 30 * normalized_t**4)
                / tf
            )
            qt_ddot[i] = (
                delta
                * (60 * normalized_t - 180 * normalized_t**2 + 120 * normalized_t**3)
                / (tf**2)
            )

        return qt, qt_dot, qt_ddot

    def motion_plan_cartesian(
        self,
        robot_model: "RobotModel",
        tf: float,
        start_transform: np.ndarray,
        target_transform: np.ndarray,
        q_init: np.ndarray,
        num_samples: int,
        spline_num_knots: int | None = None,
        ik_solver: Optional[IKSolver] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """规划笛卡尔末端位姿轨迹，并输出对应的平滑关节轨迹序列。

        Args:
            robot_model: 提供逆运动学求解的机器人模型。
            tf: 轨迹总时长，单位为秒。
            start_transform: 起始末端位姿，4x4 齐次变换矩阵。
            target_transform: 目标末端位姿，4x4 齐次变换矩阵。
            q_init: 第一个采样点的关节初值，形状为 (7,)。
            num_samples: 采样点数，至少为 2。
            spline_num_knots: 样条拟合使用的关节轨迹节点数；为 None 时自动选择。
            ik_solver: 可选 IK 函数，签名为 ``ik_solver(transform, q_seed)``。

        Returns:
            q_traj: 关节角轨迹，形状为 (num_samples, 7)。
            qd_traj: 关节速度轨迹，形状为 (num_samples, 7)。
            qdd_traj: 关节加速度轨迹，形状为 (num_samples, 7)。
        """
        if num_samples < 2:
            raise ValueError("num_samples 必须大于等于 2。")
        if tf <= 0.0:
            raise ValueError("tf 必须大于 0。")

        start_tf = np.asarray(start_transform, dtype=np.float64).reshape((4, 4))
        target_tf = np.asarray(target_transform, dtype=np.float64).reshape((4, 4))
        q_seed = np.asarray(q_init, dtype=np.float64).reshape((7,)).copy()

        output_time_samples = np.linspace(0.0, tf, num_samples, dtype=np.float64)
        if spline_num_knots is None:
            ik_num_samples = min(num_samples, max(8, int(np.ceil(tf * 20.0)) + 1))
        else:
            ik_num_samples = min(num_samples, max(2, int(spline_num_knots)))

        ik_time_samples = np.linspace(0.0, tf, ik_num_samples, dtype=np.float64)
        s, _, _ = self._quintic_scalars(ik_time_samples, tf)

        p0 = start_tf[:3, 3]
        pf = target_tf[:3, 3]
        q0 = self._rotation_matrix_to_quaternion(start_tf[:3, :3])
        qf = self._rotation_matrix_to_quaternion(target_tf[:3, :3])

        q_traj_raw = np.zeros((ik_num_samples, 7), dtype=np.float64)
        solve_ik = ik_solver or robot_model.inverse_kinematics

        for i, alpha in enumerate(s):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = p0 + alpha * (pf - p0)

            quat = self._slerp(q0, qf, float(alpha))
            transform[:3, :3] = self._quaternion_to_rotation_matrix(quat)

            q_solution = solve_ik(transform, q_seed)
            q_traj_raw[i] = q_solution
            q_seed = q_solution

        self._build_joint_splines(ik_time_samples, q_traj_raw, tf, None)
        q_traj, qd_traj, qdd_traj = self.evaluate_joint_trajectory_array(
            output_time_samples
        )
        return q_traj, qd_traj, qdd_traj

    def evaluate_joint_trajectory(
        self, t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据当前时刻解析返回平滑关节轨迹的 q、qd、qdd。"""
        if not self.joint_position_splines:
            raise ValueError("关节样条轨迹尚未构建，请先调用 motion_plan_cartesian。")

        t_clamped = float(np.clip(t, 0.0, self.trajectory_duration))
        q = np.array([float(spline(t_clamped)) for spline in self.joint_position_splines])
        qd = np.array([float(spline(t_clamped)) for spline in self.joint_velocity_splines])
        qdd = np.array([float(spline(t_clamped)) for spline in self.joint_acceleration_splines])
        return q, qd, qdd

    def evaluate_joint_trajectory_array(
        self, time_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在一组时刻上批量求值平滑关节轨迹。"""
        if not self.joint_position_splines:
            raise ValueError("关节样条轨迹尚未构建，请先调用 motion_plan_cartesian。")

        times = np.asarray(time_samples, dtype=np.float64)
        q = np.column_stack([spline(times) for spline in self.joint_position_splines])
        qd = np.column_stack([spline(times) for spline in self.joint_velocity_splines])
        qdd = np.column_stack([spline(times) for spline in self.joint_acceleration_splines])
        return q, qd, qdd

    def _build_joint_splines(
        self,
        time_samples: np.ndarray,
        q_traj: np.ndarray,
        tf: float,
        spline_num_knots: int | None,
    ) -> None:
        """基于离散关节轨迹构建每个关节的五次样条。"""
        self.trajectory_duration = float(tf)
        num_samples = len(time_samples)
        num_joints = q_traj.shape[1]

        if spline_num_knots is None:
            spline_num_knots = min(num_samples, max(8, int(np.ceil(tf * 20.0)) + 1))
        spline_num_knots = max(6, min(int(spline_num_knots), num_samples))

        knot_indices = np.linspace(0, num_samples - 1, spline_num_knots, dtype=np.int32)
        knot_indices = np.unique(knot_indices)
        knot_times = time_samples[knot_indices]
        knot_q = q_traj[knot_indices]

        spline_degree = min(5, len(knot_times) - 1)
        self.joint_position_splines = []
        self.joint_velocity_splines = []
        self.joint_acceleration_splines = []

        for joint_idx in range(num_joints):
            spline = make_interp_spline(
                knot_times,
                knot_q[:, joint_idx],
                k=spline_degree,
            )
            self.joint_position_splines.append(spline)
            self.joint_velocity_splines.append(spline.derivative(1))
            self.joint_acceleration_splines.append(spline.derivative(2))

    def _quintic_scalars(
        self, time_samples: np.ndarray, tf: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算五次多项式时间标量及其一二阶导。"""
        tau = np.clip(time_samples / tf, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / tf
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (tf**2)
        return s, s_dot, s_ddot

    def _slerp(self, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        """对两个四元数做球面线性插值，四元数顺序为 (w, x, y, z)。"""
        q0_norm = q0 / np.linalg.norm(q0)
        q1_norm = q1 / np.linalg.norm(q1)

        dot = float(np.dot(q0_norm, q1_norm))
        if dot < 0.0:
            q1_norm = -q1_norm
            dot = -dot

        if dot > 0.9995:
            result = q0_norm + alpha * (q1_norm - q0_norm)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = alpha * theta_0
        sin_theta = np.sin(theta)

        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0_norm + s1 * q1_norm

    def _rotation_matrix_to_quaternion(self, rotation: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为四元数，顺序为 (w, x, y, z)。"""
        r = np.asarray(rotation, dtype=np.float64).reshape((3, 3))
        trace = float(np.trace(r))

        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s

        quaternion = np.array([w, x, y, z], dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)

    def _quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """将四元数转换为旋转矩阵，四元数顺序为 (w, x, y, z)。"""
        w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape((4,))
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
