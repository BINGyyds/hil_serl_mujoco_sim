# data_manager.py

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

class DataManager:
    """
    负责在仿真过程中收集、存储、保存和可视化所有相关数据。

    该类作为一个数据缓冲区，将仿真每个时间点的数据（如关节角度、力矩、
    基座姿态等）收集起来，并在仿真结束后将它们一次性写入CSV文件或
    使用matplotlib进行绘图。
    """
    def __init__(self) -> None:
        """初始化数据管理器，创建用于存储时序数据的空列表。"""
        self.timestamps: List[float] = []
        self.joint_angles: List[np.ndarray] = []
        self.joint_torques: List[np.ndarray] = []
        self.end_effector_poses: List[np.ndarray] = []
        logger.info("数据管理器已初始化。")

    def record_data(self, time: float, q: np.ndarray, torques: np.ndarray, ee_pos: np.ndarray) -> None:
        """
        记录一个仿真时间点上的所有关键物理量。

        Args:
            time (float): 当前时间戳。
            q (np.ndarray): 7个关节的角度。
            torques (np.ndarray): 7个关节的力矩。
            ee_pos (np.ndarray): 末端执行器的位置 (x, y, z)。
        """
        self.timestamps.append(time)
        self.joint_angles.append(q.copy())
        self.joint_torques.append(torques.copy())
        self.end_effector_poses.append(ee_pos.copy())

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        将四元数 (w, x, y, z) 转换为XYZ顺序的欧拉角 (Roll, Pitch, Yaw)。

        Args:
            quat (np.ndarray): 格式为 [w, x, y, z] 的四元数。

        Returns:
            np.ndarray: 格式为 [roll, pitch, yaw] 的欧拉角数组。
        """
        w, x, y, z = quat
        # Roll (绕x轴旋转)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        # Pitch (绕y轴旋转)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0) # 防止因浮点数误差导致asin输入越界
        pitch_y = np.arcsin(t2)
        # Yaw (绕z轴旋转)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        return np.array([roll_x, pitch_y, yaw_z])

    def save_data(self, robot_path: Path) -> None:
        """
        将所有收集到的数据分别保存到基座和机器人的CSV文件中。
        """
        if not self.timestamps:
            logger.warning("数据缓冲区为空，没有数据可供保存。")
            return

        try:
            import pandas as pd
        except ImportError:
            logger.warning("未安装 pandas，跳过 CSV 数据保存。")
            return

        try:
            self._save_robot_data_to_csv(robot_path, pd)
        except Exception as e:
            logger.exception(f"保存数据到CSV文件时发生错误: {e}")

    def _save_robot_data_to_csv(self, file_path: Path, pd):
        """保存机器人关节和末端相关数据到指定的CSV文件。"""
        logger.info(f"正在将机器人数据保存到CSV文件: {file_path}")
        
        num_joints = len(self.joint_angles[0])

        # 1. 动态生成表头和单位
        headers = (
            ['Time'] +
            [f'J{i+1}_Angle' for i in range(num_joints)] +
            [f'J{i+1}_Torque' for i in range(num_joints)] +
            ['EE_Pos_X', 'EE_Pos_Y', 'EE_Pos_Z']
        )
        units = (
            ['s'] +
            ['rad'] * num_joints +
            ['N.m'] * num_joints +
            ['m', 'm', 'm']
        )

        # 2. 准备数据
        data_to_save = np.hstack([
            np.array(self.timestamps)[:, np.newaxis],
            np.vstack(self.joint_angles),
            np.vstack(self.joint_torques),
            np.vstack(self.end_effector_poses)
        ])

        # 3. 写入文件（两步法）
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 步骤一: 写入表头和单位行
        df_units = pd.DataFrame([units], columns=headers)
        df_units.to_csv(file_path, index=False, encoding='utf-8')
        
        # 步骤二: 以追加模式写入实际数据，不带表头
        df_data = pd.DataFrame(data=data_to_save, columns=headers)
        df_data.to_csv(file_path, mode='a', header=False, index=False, 
                       float_format='%.8f', encoding='utf-8')
        
        logger.info(f"机器人数据已成功保存到 {file_path}")

    def plot_data(self) -> None:
        """
        使用Matplotlib可视化所有已记录的数据，用于快速分析和调试。
        """
        if not self.timestamps:
            logger.warning("数据缓冲区为空，没有数据可供绘制。")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("未安装 matplotlib，跳过数据绘图。")
            return

        logger.info("正在绘制仿真数据图表...")
        num_joints = len(self.joint_angles[0])

        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        fig.suptitle("Simulation Results", fontsize=16)

        # 绘制关节角度
        axes[0].set_title("Joint Angles (q)")
        for i in range(num_joints):
            axes[0].plot(self.timestamps, [q[i] for q in self.joint_angles], label=f'Joint {i+1}')
        axes[0].set_ylabel("Angle (rad)")
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制关节力矩
        axes[1].set_title("Joint Torques (τ)")
        for i in range(num_joints):
            axes[1].plot(self.timestamps, [tau[i] for tau in self.joint_torques], label=f'Joint {i+1}')
        axes[1].set_ylabel("Torque (N·m)")
        axes[1].legend()
        axes[1].grid(True)
        
        # 绘制末端执行器位置
        axes[2].set_title("End Effector Position")
        ee_pos_arr = np.array(self.end_effector_poses)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            axes[2].plot(self.timestamps, ee_pos_arr[:, i], label=axis)
        axes[2].set_ylabel("Position (m)")
        axes[2].legend()
        axes[2].grid(True)

        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
