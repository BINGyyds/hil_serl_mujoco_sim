# robot_controller.py

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

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
        self.spacemouse: Optional[Any] = None
        self.filtered_spacemouse_action = np.zeros(6, dtype=np.float64)
        self.reset_button_pressed = False
        self.q_home = np.zeros(7, dtype=np.float64)
        self.teleop_target_transform: Optional[np.ndarray] = None
        self.teleop_posture_reference = np.zeros(7, dtype=np.float64)
        
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
        setup_started = False
        try:
            self._setup_simulation()
            setup_started = True
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
        finally:
            if setup_started or self.video_writers or self.spacemouse is not None:
                self._cleanup_simulation()

    def _setup_simulation(self):
        """
        执行所有仿真实时循环开始前的准备工作。
        包括初始化渲染器、视频写入器、数据记录计时器和查看器。
        """
        logger.info("正在准备仿真环境...")

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
        self.mujoco_interface.forward()
        self.q_home = self.mujoco_interface.get_arm_joint_angles()
        if self.config.INPUT_MODE == "planned":
            self._initialize_planned_mode()
        elif self.config.INPUT_MODE == "spacemouse":
            self._initialize_spacemouse_mode()
        else:
            raise ValueError(f"不支持的输入模式: {self.config.INPUT_MODE}")

        # 6. 初始化渲染器和视频写入器
        frame_size = None
        if self.config.MAIN_VIDEO_PATH:
            try:
                renderer = self.mujoco_interface.setup_renderer(width=1280, height=720)
                frame_size = (renderer.width, renderer.height)
            except Exception:
                logger.exception("离线渲染器初始化失败，已禁用视频录制。")
                frame_size = None

        self.video_writers = []
        if self.config.MAIN_VIDEO_PATH and frame_size is not None:
            try:
                main_writer = AsyncVideoWriter(
                    self.config.MAIN_VIDEO_PATH,
                    self.config.VIDEO_FPS,
                    frame_size,
                    self.config.VIDEO_SPEED_MULTIPLIER,
                )
                self.video_writers.append(main_writer)
            except Exception:
                logger.exception("视频写入器初始化失败，已禁用视频录制。")

        for writer in self.video_writers:
            writer.start()

        # 7. 启动被动查看器（如果需要）
        if self.config.USE_VIEWER: 
            self.mujoco_interface.launch_viewer()

    def _initialize_planned_mode(self) -> None:
        """初始化离线规划模式。"""
        self.ee_tf_home = self.mujoco_interface.get_attachment_transform()
        self.ee_tf_final = self.ee_tf_home.copy()
        self.ee_tf_final[2, 3] = 0.1

        self.q_final = self.robot_model.inverse_kinematics(
            self.ee_tf_final,
            q_init=self.q_home,
        )

        self.mujoco_interface.step()

        self.q_traj, self.qd_traj, self.qdd_traj = self.robot_motion.motion_plan_cartesian(
            robot_model=self.robot_model,
            tf=self.config.SIMULATION_DURATION,
            start_transform=self.ee_tf_home,
            target_transform=self.ee_tf_final,
            q_init=self.q_home,
            num_samples=int(self.config.SIMULATION_DURATION / self.config.TIMESTEP) + 1,
        )

    def _initialize_spacemouse_mode(self) -> None:
        """初始化 SpaceMouse teleop 模式。"""
        from robot_controllers_mujoco.spacemouse.spacemouse_expert import (
            SpaceMouseExpert,
            ZeroSpaceMouseExpert,
        )

        try:
            self.spacemouse = SpaceMouseExpert()
            logger.info("SpaceMouse teleop 模式已启用。")
        except RuntimeError as exc:
            if not self.config.ALLOW_MISSING_SPACEMOUSE:
                raise

            self.spacemouse = ZeroSpaceMouseExpert()
            logger.warning("SpaceMouse 不可用，已回退到零输入 teleop 模式: %s", exc)

        self.filtered_spacemouse_action[:] = 0.0
        self.reset_button_pressed = False
        self._sync_teleop_reference_to_current_state()

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

        if self.spacemouse is not None:
            self.spacemouse.close()
            self.spacemouse = None

        # 3. 保存记录的数据到CSV
        if self.config.ROBOT_DATA_PATH:
            self.data_manager.save_data(
                robot_path=self.config.ROBOT_DATA_PATH
            )
            
        # 4. 绘制数据图表
        self.data_manager.plot_data()

    def _handle_rendering(self, current_sim_time: float):
        """在仿真循环中处理视频帧的渲染和分发。"""
        if current_sim_time >= self.next_frame_sim_time and self.video_writers:
            # 渲染主摄像头画面
            main_frame = self.mujoco_interface.render_frame(camera=self.mujoco_interface.camera)
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
        current_q, current_v = self._get_current_joint_state()
        
        if self.config.INPUT_MODE == "planned":
            target_q, target_v, target_a = self.robot_motion.evaluate_joint_trajectory(current_sim_time)
            self._apply_control(current_q, current_v, target_q, target_v, target_a)
        elif self.config.INPUT_MODE == "spacemouse":
            joint_velocity_command = self._run_spacemouse_teleop_step(  # 更新 teleop 目标位姿
                current_q=current_q,
                current_v=current_v,
            )   
            self._send_joint_velocity_command_to_mujoco(joint_velocity_command)
        else:
            raise ValueError(f"不支持的输入模式: {self.config.INPUT_MODE}")
        
        # 3. 执行一步物理仿真
        self.mujoco_interface.step()

    def _get_current_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """读取当前关节角度和关节角速度。"""
        return self.mujoco_interface.get_arm_joint_state()

    def _get_current_end_effector_kinematics(
        self,
        current_q: np.ndarray,
        current_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用 Pinocchio 计算当前末端位姿与 LOCAL twist。"""
        return self.robot_model.end_effector_kinematics(current_q, current_v)

    def _sync_teleop_reference_to_current_state(self) -> None:
        """将 teleop 初始期望位姿同步到 MuJoCo 当前状态。"""
        self.mujoco_interface.forward()
        current_q, _ = self._get_current_joint_state()
        self.teleop_target_transform = self.mujoco_interface.get_attachment_transform()
        self.teleop_posture_reference = current_q.copy()
        self.filtered_spacemouse_action[:] = 0.0

    def _read_spacemouse_input(self) -> Tuple[np.ndarray, list]:
        """读取最新的 SpaceMouse 6D 输入和按钮状态。"""
        if self.spacemouse is None:
            raise RuntimeError("SpaceMouse 尚未初始化。")

        action, buttons = self.spacemouse.get_action()
        return np.asarray(action[:6], dtype=np.float64), buttons

    def _handle_spacemouse_reset_request(self, buttons: list) -> bool:
        """处理 SpaceMouse 的复位请求。"""
        reset_pressed = bool(buttons[0]) if buttons else False
        if reset_pressed:
            if not self.reset_button_pressed:
                self.mujoco_interface.reset_to_home_keyframe()
                self._sync_teleop_reference_to_current_state()
            self.reset_button_pressed = True
            return True

        self.reset_button_pressed = reset_pressed
        return False

    def _run_spacemouse_teleop_step(
        self,
        current_q: np.ndarray,
        current_v: np.ndarray,
    ) -> np.ndarray:
        """执行一次 SpaceMouse teleop 控制计算。"""
        raw_spacemouse_action, buttons = self._read_spacemouse_input()
        if self._handle_spacemouse_reset_request(buttons):
            return np.zeros_like(current_q)

        current_transform, _ = self._get_current_end_effector_kinematics(
            current_q,
            current_v,
        )
        desired_transform = self._update_desired_transform_from_spacemouse(
            raw_spacemouse_action
        )
        return self._compute_joint_velocity_command_from_pose_error(
            current_q=current_q,
            current_transform=current_transform,
            desired_transform=desired_transform,
        )

    def _update_desired_transform_from_spacemouse(
        self,
        raw_spacemouse_action: np.ndarray,
    ) -> np.ndarray:
        """根据 SpaceMouse 增量输入更新期望末端位姿。"""
        if self.teleop_target_transform is None:
            self._sync_teleop_reference_to_current_state()

        twist_command = self._filter_spacemouse_action(raw_spacemouse_action)
        self._integrate_teleop_target_transform(twist_command)
        return self.teleop_target_transform.copy()

    def _compute_joint_velocity_command_from_pose_error(
        self,
        current_q: np.ndarray,
        current_transform: np.ndarray,
        desired_transform: np.ndarray,
    ) -> np.ndarray:
        """根据当前/期望末端位姿误差计算关节命令速度。"""
        pose_error = self._compute_local_pose_error(
            current_transform=current_transform,
            target_transform=desired_transform,
        )

        K_pose = 25.0  # 末端空间的比例增益 

        jacobian = self.robot_model.jacobian(current_q)
        damping_matrix = self.config.IK_DAMPING * np.eye(6, dtype=np.float64)
        jjt = jacobian @ jacobian.T
        jacobian_pinv = jacobian.T @ np.linalg.solve(jjt + damping_matrix, np.eye(6))
        qdot_task = K_pose * jacobian_pinv @ pose_error

        qdot_null = self.config.NULLSPACE_GAIN * (self.teleop_posture_reference - current_q)
        nullspace_projector = np.eye(jacobian.shape[1], dtype=np.float64) - jacobian_pinv @ jacobian
        qdot_command = qdot_task + nullspace_projector @ qdot_null
        qdot_command = self._apply_joint_limit_guard(current_q, qdot_command)

        joint_velocity_limits = np.asarray(
            self.config.JOINT_VELOCITY_LIMITS,
            dtype=np.float64,
        )
        qdot_command = np.clip(qdot_command, -joint_velocity_limits, joint_velocity_limits)
        return qdot_command

    def _send_joint_velocity_command_to_mujoco(
        self,
        joint_velocity_command: np.ndarray,
    ) -> None:
        """将关节速度命令按关节名映射后下发到 MuJoCo。"""
        joint_velocity = np.asarray(joint_velocity_command, dtype=np.float64)
        if joint_velocity.shape != (len(self.config.MUJOCO_ARM_JOINT_NAMES),):
            raise ValueError(
                "关节速度命令维度错误，"
                f"期望 {(len(self.config.MUJOCO_ARM_JOINT_NAMES),)}，"
                f"实际 {joint_velocity.shape}"
            )

        joint_velocity_by_name: Dict[str, float] = {
            joint_name: float(velocity)
            for joint_name, velocity in zip(
                self.config.MUJOCO_ARM_JOINT_NAMES,
                joint_velocity,
            )
        }
        self.mujoco_interface.set_named_joint_velocity_command(joint_velocity_by_name)

    def _filter_spacemouse_action(self, raw_action: np.ndarray) -> np.ndarray:
        """对 SpaceMouse 输入做死区、缩放和低通滤波。"""
        if raw_action.shape != (6,):
            raise ValueError(f"SpaceMouse 输入维度错误，期望 (6,)，实际 {raw_action.shape}")

        action = raw_action.copy()
        action[np.abs(action) < self.config.SPACEMOUSE_DEADZONE] = 0.0
        if not np.any(action):
            self.filtered_spacemouse_action[:] = 0.0
            return np.zeros(6, dtype=np.float64)

        scaled_action = action.copy()
        scaled_action[:3] *= self.config.SPACEMOUSE_TRANSLATION_SCALE
        scaled_action[3:] *= self.config.SPACEMOUSE_ROTATION_SCALE

        alpha = float(np.clip(self.config.SPACEMOUSE_FILTER_ALPHA, 0.0, 1.0))
        self.filtered_spacemouse_action = (
            alpha * scaled_action + (1.0 - alpha) * self.filtered_spacemouse_action
        )
        return self.filtered_spacemouse_action.copy()

    def _integrate_teleop_target_transform(self, twist_command: np.ndarray) -> None:
        """将 SpaceMouse 增量命令积分到末端参考位姿上。"""
        if self.teleop_target_transform is None:
            raise RuntimeError("teleop 参考位姿尚未初始化。")

        dt = float(self.config.TIMESTEP)
        delta_translation_local = twist_command[:3] * dt
        delta_rotation_local = twist_command[3:] * dt

        target_transform = self.teleop_target_transform.copy()
        target_rotation = target_transform[:3, :3]
        target_position = target_transform[:3, 3]

        target_position = target_position + delta_translation_local
        target_rotation = Rotation.from_rotvec(delta_rotation_local).as_matrix() @ target_rotation

        self.teleop_target_transform[:3, :3] = target_rotation
        self.teleop_target_transform[:3, 3] = target_position

    def _compute_local_pose_error(
        self,
        current_transform: np.ndarray,
        target_transform: np.ndarray,
    ) -> np.ndarray:
        """使用 Pinocchio log6 计算 ``log6(T * T_d^{-1})`` 形式的 6D 位姿误差。"""
        current_pose = pin.SE3(
            np.asarray(current_transform[:3, :3], dtype=np.float64),
            np.asarray(current_transform[:3, 3], dtype=np.float64),
        )
        target_pose = pin.SE3(
            np.asarray(target_transform[:3, :3], dtype=np.float64),
            np.asarray(target_transform[:3, 3], dtype=np.float64),
        )
        pose_error = current_pose.inverse() * target_pose
        return np.asarray(pin.log6(pose_error).vector, dtype=np.float64).reshape(6)

    def _apply_joint_limit_guard(
        self,
        current_q: np.ndarray,
        qdot_command: np.ndarray,
    ) -> np.ndarray:
        """在接近关节限位时抑制继续向外的速度命令。"""
        qdot_safe = np.asarray(qdot_command, dtype=np.float64).copy()
        lower = self.robot_model.lower_position_limits
        upper = self.robot_model.upper_position_limits
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        margin = self.config.JOINT_LIMIT_MARGIN

        lower_block = finite_mask & (current_q <= lower + margin) & (qdot_safe < 0.0)
        upper_block = finite_mask & (current_q >= upper - margin) & (qdot_safe > 0.0)
        qdot_safe[lower_block | upper_block] = 0.0
        return qdot_safe

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
