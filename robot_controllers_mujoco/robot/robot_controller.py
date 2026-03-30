# robot_controller.py

import logging
from typing import List

import numpy as np

from robot_controllers_mujoco.robot.async_writer import AsyncVideoWriter
from robot_controllers_mujoco.robot.config import ProjectConfig
from robot_controllers_mujoco.robot.data_manager import DataManager
from robot_controllers_mujoco.robot.mujoco_interface import MujocoInterface
from robot_controllers_mujoco.robot.robot_model import RobotModel
from robot_controllers_mujoco.robot.robot_motion import RobotMotion

logger = logging.getLogger(__name__)

class RobotController:
    """
    主控制框架，负责协调所有模块并执行高层仿真与控制逻辑。

    该类是整个项目的“大脑”，它将底层的MuJoCo接口、动力学模型、轨迹规划器
    和数据管理器整合在一起，执行完整的仿真任务。
    """
    def __init__(self, config: ProjectConfig) -> None:
        """
        初始化机器人控制器。

        Args:
            config (ProjectConfig): 包含所有配置参数的全局配置对象。
        """
        logger.info("正在初始化机器人控制器...")
        self.config = config
        self.mujoco_interface = MujocoInterface(config)
        self.robot_model = RobotModel(mjcf_path=config.MJCF_PATH, end_effector_frame_name=config.MUJOCO_END_EFFECTOR_SITE_NAME)
        self.robot_motion = RobotMotion()
        self.data_manager = DataManager()
        
        # --- 内部状态变量 ---
        self.video_writers: List[AsyncVideoWriter] = []

        # --- 控制器增益 ---
        self.EFFORT_KP = np.diag(config.EFFORT_KP_GAINS)
        self.EFFORT_KD = np.diag(config.EFFORT_KD_GAINS)

    def run_simulation(self) -> None:
        """
        运行主仿真循环。
        该循环以物理仿真的最高可能速度运行，并根据需要处理渲染、数据记录和控制逻辑。
        """
        self._setup_simulation()
        logger.info("主仿真循环开始...")

        while self.mujoco_interface.data.time < self.config.SIMULATION_DURATION:
            current_sim_time = self.mujoco_interface.data.time

            # 模块化处理循环内的各项任务
            # self._handle_rendering(current_sim_time)
            self._handle_data_logging(current_sim_time)
            self._execute_simulation_step(current_sim_time)
            
            # 与查看器同步（如果开启）
            if self.config.USE_VIEWER: 
                self.mujoco_interface.sync_viewer()

        logger.info("主仿真循环结束。")
        self._cleanup_simulation()

    def _setup_simulation(self):
        """
        执行所有仿真实时循环开始前的准备工作。
        包括初始化渲染器、视频写入器、数据记录计时器和查看器。
        """
        logger.info("正在准备仿真环境...")
        # 1. 初始化离线渲染器以确定帧尺寸
        renderer = self.mujoco_interface.setup_renderer(width=1280, height=720)
        frame_size = (renderer.width, renderer.height)
        
        # 2. 根据配置创建并启动所有异步视频写入器
        self.video_writers = []
        if self.config.MAIN_VIDEO_PATH:
            main_writer = AsyncVideoWriter(self.config.MAIN_VIDEO_PATH, self.config.VIDEO_FPS, frame_size, self.config.VIDEO_SPEED_MULTIPLIER)
            self.video_writers.append(main_writer)

        for writer in self.video_writers:
            writer.start()

        # 3. 初始化仿真循环内的计时器
        self.video_frame_duration = 1.0 / self.config.VIDEO_FPS
        self.next_frame_sim_time = 0.0
        self.next_log_time = 0.0

        # 4. 设置默认相机视角
        self.mujoco_interface.set_camera_view(
            azimuth=72.91, elevation=-49.89, distance=16.26, lookat=np.array([-0.22, 0.08, 1.66])
        )

        # 5. 在启动 viewer 前将机械臂重置到 home 位置。
        self.mujoco_interface.reset_to_home_keyframe()
        
        self.ee_tf_home = self.mujoco_interface.get_attachment_transform()
        self.ee_tf_final = self.ee_tf_home.copy()
        # self.ee_tf_final[:3, 3] += np.array([0.0, 0.0, -0.55])  # 在 z 方向向下平移 0.55 米
        self.ee_tf_final[2, 3] = 0.1  # z 方向目标高度为 0.1 米

        self.q_home = self.mujoco_interface.get_arm_joint_angles()
        self.q_final = self.robot_model.inverse_kinematics(self.ee_tf_final, q_init=self.q_home)

        self.mujoco_interface.step()

        self.q_traj, self.qd_traj, self.qdd_traj = self.robot_motion.motion_plan_cartesian(
            robot_model=self.robot_model,
            tf=self.config.SIMULATION_DURATION,
            start_transform=self.ee_tf_home,
            target_transform=self.ee_tf_final,
            q_init=self.q_home,
            num_samples=int(self.config.SIMULATION_DURATION / self.config.TIMESTEP) + 1,
        )

        # 6. 启动被动查看器（如果需要）
        if self.config.USE_VIEWER: 
            self.mujoco_interface.launch_viewer()

    def _cleanup_simulation(self):
        """
        执行所有仿真结束后的清理工作。
        包括停止视频写入器、关闭查看器、保存数据等。
        """
        logger.info("正在清理仿真资源...")
        # 1. 停止所有异步写入器，确保视频文件被正确关闭
        for writer in self.video_writers:
            writer.stop()
        self.video_writers = [] # 清空列表
            
        # 2. 关闭查看器
        if self.config.USE_VIEWER:
            self.mujoco_interface.close_viewer()

        # 3. 保存记录的数据到CSV
        if self.config.ROBOT_DATA_PATH:
            self.data_manager.save_data(
                robot_path=self.config.ROBOT_DATA_PATH
            )
            
        # 4. 绘制数据图表
        self.data_manager.plot_data()

    def _handle_rendering(self, current_sim_time: float):
        """在仿真循环中处理视频帧的渲染和分发。"""
        if current_sim_time >= self.next_frame_sim_time:
            # 渲染主摄像头画面
            main_frame = self.mujoco_interface.render_frame(camera=self.mujoco_interface.camera)
            if self.config.MAIN_VIDEO_PATH:
                self.video_writers[0].add_frame(main_frame)
            # 更新下一个渲染时间点
            self.next_frame_sim_time += self.video_frame_duration

    def _handle_data_logging(self, current_sim_time: float):
        """在仿真循环中处理数据的采样和记录。"""
        if current_sim_time >= self.next_log_time:
            # 获取所有需要记录的数据
            current_q = self.mujoco_interface.get_arm_joint_angles()
            torques = self.mujoco_interface.get_joint_torques()
            ee_pos = self.mujoco_interface.get_end_effector_position()

            # 打印关键信息到控制台
            if current_sim_time % 2.0 < self.config.TIMESTEP:  # 每2秒打印一次
                self.print_robot_info(ee_pos)
            
            # 将数据传递给数据管理器
            self.data_manager.record_data(self.next_log_time, current_q, torques, ee_pos)
            
            self.next_log_time += self.config.DATA_LOGGING_INTERVAL

    def _execute_simulation_step(self, current_sim_time: float):
        """执行单步的物理仿真，包括获取状态、计算控制指令和步进。"""
        # 1. 获取当前机器人状态
        current_q = self.mujoco_interface.get_arm_joint_angles()
        current_v = self.mujoco_interface.get_arm_joint_velocities()
        
        # 2. 从轨迹规划器获取当前时刻的目标状态
        target_q, target_v, target_a = self.robot_motion.evaluate_joint_trajectory(
            current_sim_time
        )

        self._apply_control(current_q, current_v, target_q, target_v, target_a)
        
        # 3. 执行一步物理仿真
        self.mujoco_interface.step()

    def _apply_control(self, current_q: np.ndarray, current_v: np.ndarray, target_q: np.ndarray, target_v: np.ndarray, target_a: np.ndarray) -> None:
        """
        根据当前控制模式计算并应用控制指令到仿真器。
        
        Args:
            current_q: 当前关节角度。
            current_v: 当前关节速度。
            target_q: 目标关节角度。
            target_v: 目标关节速度。
            target_a: 目标关节加速度。
        """

        if self.config.CONTROL_MODE == 'position':
            # 在位置模式下，直接将目标关节角度作为控制信号。
            control_signal = target_q
        elif self.config.CONTROL_MODE == 'effort':
            # 在力矩模式下，使用前馈+反馈（PD）控制律。
            # 1. 动力学前馈 (Inverse Dynamics)
            tau_ff = self.robot_model.inverse_dynamics(current_q, current_v, target_a)
            # tau_ff = self.robot_model.inverse_dynamics(target_q, target_v, target_a)
            # 2. PD反馈
            q_error = target_q - current_q
            v_error = target_v - current_v
            tau_fb = self.EFFORT_KP @ q_error + self.EFFORT_KD @ v_error
            # 3. 总控制力矩
            control_signal = tau_ff + tau_fb
        else:
            raise ValueError(f"不支持的控制模式: {self.config.CONTROL_MODE}")
        
        self.mujoco_interface.set_control(control_signal)

    def print_robot_info(self, ee_pos) -> None:
        """打印机器人和仿真器的关键信息。"""
        logger.info("=" * 60)
        logger.info("机器人信息:")
        
        logger.info(f"时间步长: {self.config.TIMESTEP}s")
        logger.info(f"末端位置:{ee_pos}")

        logger.info("=" * 60)
