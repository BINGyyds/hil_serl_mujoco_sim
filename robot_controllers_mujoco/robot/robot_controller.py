import logging
import time
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
    """协调 MuJoCo、Pinocchio 与输入设备的高层控制器。"""

    def __init__(self, config: ProjectConfig) -> None:
        logger.info("正在初始化机器人控制器...")
        self.config = config
        self.mujoco_interface = MujocoInterface(config)
        self.robot_model = RobotModel(
            mjcf_path=config.model.mjcf_path,
            end_effector_frame_name=config.model.end_effector_site_name,
        )
        self.robot_motion = RobotMotion()
        self.data_manager = DataManager()

        self.spacemouse: Optional[Any] = None
        self.video_writers: List[AsyncVideoWriter] = []

        self.filtered_spacemouse_action = np.zeros(6, dtype=np.float64)
        self.reset_button_pressed = False
        self.q_home = np.zeros(7, dtype=np.float64)
        self.teleop_target_transform: Optional[np.ndarray] = None
        self.teleop_posture_reference = np.zeros(7, dtype=np.float64)

        self.wall_time_start = 0.0
        self.sim_time_start = 0.0
        self.video_frame_duration = 0.0
        self.next_frame_sim_time = 0.0
        self.next_log_time = 0.0

        self.effort_kp = np.diag(config.effort_control.kp_gains)
        self.effort_kd = np.diag(config.effort_control.kd_gains)

    def run_simulation(self) -> None:
        """运行主仿真循环。"""
        setup_started = False
        try:
            self._setup_simulation()
            setup_started = True
            logger.info("主仿真循环开始...")

            while True:
                current_sim_time = self.mujoco_interface.data.time
                self._handle_data_logging(current_sim_time)
                self._execute_simulation_step(current_sim_time)

                updated_sim_time = self.mujoco_interface.data.time
                self._handle_rendering(updated_sim_time)
                self._sync_to_real_time(updated_sim_time)
        finally:
            if setup_started or self.video_writers or self.spacemouse is not None:
                self._cleanup_simulation()

    def _setup_simulation(self) -> None:
        """执行主循环开始前的准备工作。"""
        logger.info("正在准备仿真环境...")

        self.video_frame_duration = 1.0 / self.config.visualization.video_fps
        self.next_frame_sim_time = self.mujoco_interface.data.time
        self.next_log_time = 0.0
        self.wall_time_start = time.perf_counter()
        self.sim_time_start = self.mujoco_interface.data.time

        self.mujoco_interface.set_camera_view(
            azimuth=72.91,
            elevation=-49.89,
            distance=16.26,
            lookat=np.array([-0.22, 0.08, 1.66]),
        )

        self.mujoco_interface.reset_to_home_keyframe()
        self.mujoco_interface.forward()
        self.q_home = self.mujoco_interface.get_arm_joint_angles()

        if self.config.mode.input_mode == "planned":
            self._initialize_planned_mode()
        elif self.config.mode.input_mode == "spacemouse":
            self._initialize_spacemouse_mode()
        else:
            raise ValueError(f"不支持的输入模式: {self.config.mode.input_mode}")

        frame_size = None
        main_video_path = self.config.paths.main_video_path
        if main_video_path:
            try:
                renderer = self.mujoco_interface.setup_renderer(width=1280, height=720)
                frame_size = (renderer.width, renderer.height)
            except Exception:
                logger.exception("离线渲染器初始化失败，已禁用视频录制。")
                frame_size = None

        self.video_writers = []
        if main_video_path and frame_size is not None:
            try:
                main_writer = AsyncVideoWriter(
                    main_video_path,
                    self.config.visualization.video_fps,
                    frame_size,
                    self.config.visualization.video_speed_multiplier,
                )
                self.video_writers.append(main_writer)
            except Exception:
                logger.exception("视频写入器初始化失败，已禁用视频录制。")

        for writer in self.video_writers:
            writer.start()

        if self.config.visualization.use_viewer:
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

        duration = self.config.sim.simulation_duration
        timestep = self.config.sim.timestep
        self.q_traj, self.qd_traj, self.qdd_traj = self.robot_motion.motion_plan_cartesian(
            robot_model=self.robot_model,
            tf=duration,
            start_transform=self.ee_tf_home,
            target_transform=self.ee_tf_final,
            q_init=self.q_home,
            num_samples=int(duration / timestep) + 1,
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
            if not self.config.mode.allow_missing_spacemouse:
                raise

            self.spacemouse = ZeroSpaceMouseExpert()
            logger.warning("SpaceMouse 不可用，已回退到零输入 teleop 模式: %s", exc)

        self.filtered_spacemouse_action[:] = 0.0
        self.reset_button_pressed = False
        self._sync_teleop_reference_to_current_state()

    def _cleanup_simulation(self) -> None:
        """执行所有仿真结束后的清理工作。"""
        logger.info("主仿真循环结束,正在清理仿真资源...")

        for writer in self.video_writers:
            writer.stop()
        self.video_writers = []

        if self.config.visualization.use_viewer:
            self.mujoco_interface.close_viewer()

        if self.spacemouse is not None:
            self.spacemouse.close()
            self.spacemouse = None

        robot_data_path = self.config.paths.robot_data_path
        if robot_data_path:
            self.data_manager.save_data(robot_path=robot_data_path)

        self.data_manager.plot_data()

    def _handle_data_logging(self, current_sim_time: float) -> None:
        """在仿真循环中处理数据采样与日志输出。"""
        if current_sim_time >= self.next_log_time:
            current_q = self.mujoco_interface.get_arm_joint_angles()
            torques = self.mujoco_interface.get_joint_torques()
            ee_pos = self.mujoco_interface.get_end_effector_position()

            if current_sim_time % 2.0 < self.config.sim.timestep:
                self.print_robot_info(ee_pos)

            self.data_manager.record_data(self.next_log_time, current_q, torques, ee_pos)
            self.next_log_time += self.config.logging.data_logging_interval

    def _execute_simulation_step(self, current_sim_time: float) -> None:
        """执行单步控制与物理仿真。"""
        current_q, current_v = self._get_current_joint_state()

        if self.config.mode.input_mode == "planned":
            target_q, target_v, target_a = self.robot_motion.evaluate_joint_trajectory(
                current_sim_time
            )
            self._apply_control(current_q, current_v, target_q, target_v, target_a)
        elif self.config.mode.input_mode == "spacemouse":
            joint_velocity_command = self._run_spacemouse_teleop_step(
                current_q=current_q,
                current_v=current_v,
            )
            self._send_joint_velocity_command_to_mujoco(joint_velocity_command)
        else:
            raise ValueError(f"不支持的输入模式: {self.config.mode.input_mode}")

        self.mujoco_interface.step()

    def _handle_rendering(self, current_sim_time: float) -> None:
        """在仿真循环中处理 viewer 同步。"""
        # if self.video_writers and current_sim_time >= self.next_frame_sim_time:
        #     main_frame = self.mujoco_interface.render_frame(camera=self.mujoco_interface.camera)
        #     self.video_writers[0].add_frame(main_frame)
        #     while current_sim_time >= self.next_frame_sim_time:
        #         self.next_frame_sim_time += self.video_frame_duration
        if self.config.visualization.use_viewer:
            self.mujoco_interface.sync_viewer()

    def _sync_to_real_time(self, current_sim_time: float) -> None:
        """将仿真时间与墙钟时间对齐。"""
        if not self.config.sim.sync_to_real_time:
            return

        target_wall_time = self.wall_time_start + (
            (current_sim_time - self.sim_time_start) / self.config.sim.real_time_factor
        )
        remaining = target_wall_time - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    def print_robot_info(self, ee_pos: np.ndarray) -> None:
        """打印机器人和仿真器的关键信息。"""
        logger.info("=" * 60)
        logger.info("机器人信息:")
        logger.info("时间步长: %ss", self.config.sim.timestep)
        logger.info("末端位置:%s", ee_pos)
        logger.info("=" * 60)

    def _apply_control(
        self,
        current_q: np.ndarray,
        current_v: np.ndarray,
        target_q: np.ndarray,
        target_v: np.ndarray,
        target_a: np.ndarray,
    ) -> None:
        """根据 planned 模式下的目标轨迹向 MuJoCo 下发控制指令。"""
        if self.config.mode.control_mode == "position":
            control_signal = target_q
        elif self.config.mode.control_mode == "effort":
            tau_ff = self.robot_model.inverse_dynamics(current_q, current_v, target_a)
            q_error = target_q - current_q
            v_error = target_v - current_v
            tau_fb = self.effort_kp @ q_error + self.effort_kd @ v_error
            control_signal = tau_ff + tau_fb
        else:
            raise ValueError(f"不支持的控制模式: {self.config.mode.control_mode}")

        self.mujoco_interface.set_control(control_signal)

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
        """将 teleop 参考位姿同步到 MuJoCo 当前状态。"""
        self.mujoco_interface.forward()
        current_q, _ = self._get_current_joint_state()
        self.teleop_target_transform = self.mujoco_interface.get_attachment_transform()
        self.teleop_posture_reference = current_q.copy()
        self.filtered_spacemouse_action[:] = 0.0

    def _update_desired_transform_from_spacemouse(
        self,
        raw_spacemouse_action: np.ndarray,
    ) -> np.ndarray:
        """根据 SpaceMouse 输入更新期望末端位姿。"""
        if self.teleop_target_transform is None:
            self._sync_teleop_reference_to_current_state()

        filtered_command = self._filter_spacemouse_input(raw_spacemouse_action)
        self._integrate_teleop_target_transform(filtered_command)
        return self.teleop_target_transform.copy()

    def _filter_spacemouse_input(self, raw_action: np.ndarray) -> np.ndarray:
        """对 SpaceMouse 输入做死区和一阶低通滤波。"""
        if raw_action.shape != (6,):
            raise ValueError(f"SpaceMouse 输入维度错误，期望 (6,)，实际 {raw_action.shape}")

        action = raw_action.copy()
        action[
            np.abs(action) < self.config.teleop.spacemouse_deadzone
        ] = 0.0
        if not np.any(action):
            self.filtered_spacemouse_action[:] = 0.0
            return np.zeros(6, dtype=np.float64)

        alpha = float(
            np.clip(self.config.teleop.spacemouse_filter_alpha, 0.0, 1.0)
        )
        self.filtered_spacemouse_action = (
            alpha * action + (1.0 - alpha) * self.filtered_spacemouse_action
        )
        return self.filtered_spacemouse_action.copy()

    def _integrate_teleop_target_transform(self, filtered_command: np.ndarray) -> None:
        """将归一化 teleop 命令积分为末端参考位姿。"""
        if self.teleop_target_transform is None:
            raise RuntimeError("teleop 参考位姿尚未初始化。")

        dt = float(self.config.sim.timestep)
        delta_translation_local = (
            filtered_command[:3]
            * dt
            * self.config.teleop.max_reference_linear_velocity
        )
        delta_rotation_local = (
            filtered_command[3:]
            * dt
            * self.config.teleop.max_reference_angular_velocity
        )

        target_transform = self.teleop_target_transform.copy()
        target_rotation = target_transform[:3, :3]
        target_position = target_transform[:3, 3]

        target_position = target_position + delta_translation_local
        target_rotation = (
            Rotation.from_rotvec(delta_rotation_local).as_matrix() @ target_rotation
        )

        self.teleop_target_transform[:3, :3] = target_rotation
        self.teleop_target_transform[:3, 3] = target_position

    def _compute_joint_velocity_command_from_pose_error(
        self,
        current_q: np.ndarray,
        current_transform: np.ndarray,
        desired_transform: np.ndarray,
    ) -> np.ndarray:
        """根据当前/期望末端位姿误差计算关节速度命令。"""
        pose_error = self._compute_local_pose_error(
            current_transform=current_transform,
            target_transform=desired_transform,
        )

        jacobian = self.robot_model.jacobian(current_q)
        damping_matrix = self.config.teleop.ik_damping * np.eye(6, dtype=np.float64)
        jjt = jacobian @ jacobian.T
        jacobian_pinv = jacobian.T @ np.linalg.solve(
            jjt + damping_matrix,
            np.eye(6),
        )
        qdot_task = self.config.teleop.pose_error_gain * (jacobian_pinv @ pose_error)

        qdot_null = self.config.teleop.nullspace_gain * (
            self.teleop_posture_reference - current_q
        )
        nullspace_projector = (
            np.eye(jacobian.shape[1], dtype=np.float64) - jacobian_pinv @ jacobian
        )
        qdot_command = qdot_task + nullspace_projector @ qdot_null
        qdot_command = self._apply_joint_limit_guard(current_q, qdot_command)

        joint_velocity_limits = np.asarray(
            self.config.teleop.joint_velocity_limits,
            dtype=np.float64,
        )
        qdot_command = np.clip(qdot_command, -joint_velocity_limits, joint_velocity_limits)
        return qdot_command

    def _compute_local_pose_error(
        self,
        current_transform: np.ndarray,
        target_transform: np.ndarray,
    ) -> np.ndarray:
        """使用 Pinocchio log6 计算 6D LOCAL 位姿误差。"""
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
        margin = self.config.teleop.joint_limit_margin

        lower_block = finite_mask & (current_q <= lower + margin) & (qdot_safe < 0.0)
        upper_block = finite_mask & (current_q >= upper - margin) & (qdot_safe > 0.0)
        qdot_safe[lower_block | upper_block] = 0.0
        return qdot_safe

    def _send_joint_velocity_command_to_mujoco(
        self,
        joint_velocity_command: np.ndarray,
    ) -> None:
        """将关节速度命令按关节名映射后下发到 MuJoCo。"""
        joint_velocity = np.asarray(joint_velocity_command, dtype=np.float64)
        expected_shape = (len(self.config.model.arm_joint_names),)
        if joint_velocity.shape != expected_shape:
            raise ValueError(
                "关节速度命令维度错误，"
                f"期望 {expected_shape}，"
                f"实际 {joint_velocity.shape}"
            )

        joint_velocity_by_name: Dict[str, float] = {
            joint_name: float(velocity)
            for joint_name, velocity in zip(
                self.config.model.arm_joint_names,
                joint_velocity,
            )
        }
        self.mujoco_interface.set_named_joint_velocity_command(joint_velocity_by_name)
