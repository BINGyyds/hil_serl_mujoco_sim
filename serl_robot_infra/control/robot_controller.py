import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from serl_robot_infra.common.async_camera_window import AsyncCameraWindow
from serl_robot_infra.common.async_writer import AsyncVideoWriter
from serl_robot_infra.common.config import ProjectConfig
from serl_robot_infra.common.data_manager import DataManager
from serl_robot_infra.control.task_space_velocity_controller import (
    TaskSpaceVelocityController,
)
from serl_robot_infra.control.teleop_runtime import TeleopRuntime
from serl_robot_infra.simulation.mujoco_interface import MujocoInterface

logger = logging.getLogger(__name__)


class RobotController:
    """Franka insertion 的本地 SpaceMouse + velocity teleop 控制器。"""

    def __init__(self, config: ProjectConfig) -> None:
        logger.info("正在初始化机器人控制器...")
        self.config = config
        self.mujoco_interface = MujocoInterface(config)
        self.data_manager = DataManager()

        self.spacemouse: Optional[Any] = None
        self.teleop_runtime: Optional[TeleopRuntime] = None
        self.video_writers: List[AsyncVideoWriter] = []
        self.teleop_camera_window: Optional[AsyncCameraWindow] = None

        self.reset_button_pressed = False
        self.q_home = np.zeros(7, dtype=np.float64)

        self.wall_time_start = 0.0
        self.sim_time_start = 0.0
        self.video_frame_duration = 0.0
        self.next_frame_sim_time = 0.0
        self.next_log_time = 0.0

        self.task_space_controller = TaskSpaceVelocityController(
            input_deadzone=config.teleop.spacemouse_deadzone,
            input_filter_alpha=config.teleop.spacemouse_filter_alpha,
            max_reference_linear_velocity=config.teleop.max_reference_linear_velocity,
            max_reference_angular_velocity=config.teleop.max_reference_angular_velocity,
            pose_error_gain=config.teleop.pose_error_gain,
            ik_damping=config.teleop.ik_damping,
            nullspace_gain=config.teleop.nullspace_gain,
            joint_velocity_limits=np.asarray(
                config.teleop.joint_velocity_limits,
                dtype=np.float64,
            ),
            joint_limit_margin=config.teleop.joint_limit_margin,
            lower_position_limits=self.mujoco_interface.arm_joint_lower_limits,
            upper_position_limits=self.mujoco_interface.arm_joint_upper_limits,
            enable_admittance=config.teleop.enable_admittance,
            wrench_filter_alpha=config.teleop.wrench_filter_alpha,
            force_deadband=np.asarray(config.teleop.force_deadband, dtype=np.float64),
            torque_deadband=np.asarray(config.teleop.torque_deadband, dtype=np.float64),
            max_force_world=np.asarray(config.teleop.max_force_world, dtype=np.float64),
            max_torque_world=np.asarray(config.teleop.max_torque_world, dtype=np.float64),
            admittance_mass=np.asarray(config.teleop.admittance_mass, dtype=np.float64),
            admittance_damping=np.asarray(
                config.teleop.admittance_damping,
                dtype=np.float64,
            ),
            admittance_stiffness=np.asarray(
                config.teleop.admittance_stiffness,
                dtype=np.float64,
            ),
            max_compliance_linear_offset=config.teleop.max_compliance_linear_offset,
            max_compliance_angular_offset=config.teleop.max_compliance_angular_offset,
            max_compliance_linear_velocity=config.teleop.max_compliance_linear_velocity,
            max_compliance_angular_velocity=config.teleop.max_compliance_angular_velocity,
            joint_velocity_filter_alpha=config.teleop.joint_velocity_filter_alpha,
        )
        self.teleop_runtime = TeleopRuntime(
            controller=self.task_space_controller,
            get_control_state=self._get_teleop_control_state,
            set_joint_velocity=self._send_joint_velocity_command_to_mujoco,
            step_simulation=self.mujoco_interface.step,
            control_dt=config.sim.timestep,
            get_wrench_local=self.mujoco_interface.get_end_effector_wrench_local,
            tare_wrench=self.mujoco_interface.tare_end_effector_wrench,
            enable_admittance=config.teleop.enable_admittance,
        )

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
                self._execute_simulation_step()

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

        self.mujoco_interface.set_camera_view(
            azimuth=72.91,
            elevation=-49.89,
            distance=16.26,
            lookat=np.array([-0.22, 0.08, 1.66]),
        )

        self.mujoco_interface.reset_to_home_keyframe()
        self.mujoco_interface.forward()
        self._reset_loop_timing()
        self.q_home = self.mujoco_interface.get_arm_joint_angles()

        self._initialize_spacemouse_mode()

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

        self._setup_teleop_camera_window()

        if self.config.visualization.use_viewer:
            self.mujoco_interface.launch_viewer()

    def _initialize_spacemouse_mode(self) -> None:
        """初始化 SpaceMouse teleop 模式。"""
        from serl_robot_infra.spacemouse.spacemouse_expert import (
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

        self.reset_button_pressed = False
        if (
            self.config.teleop.enable_admittance
            and not self.mujoco_interface.has_end_effector_wrench_sensors()
        ):
            raise RuntimeError(
                "已启用 teleop 导纳柔顺，但当前模型未配置末端 force/torque sensors。"
            )
        self._sync_teleop_reference_to_current_state()

    def _cleanup_simulation(self) -> None:
        """执行所有仿真结束后的清理工作。"""
        logger.info("主仿真循环结束,正在清理仿真资源...")

        for writer in self.video_writers:
            writer.stop()
        self.video_writers = []

        self._close_teleop_camera_window()

        if self.config.visualization.use_viewer:
            self.mujoco_interface.close_viewer()
        self.mujoco_interface.close_renderers()

        if self.spacemouse is not None:
            self.spacemouse.close()
            self.spacemouse = None

        robot_data_path = self.config.paths.robot_data_path
        if robot_data_path:
            self.data_manager.save_data(robot_path=robot_data_path)

    def _handle_data_logging(self, current_sim_time: float) -> None:
        """在仿真循环中处理数据采样与日志输出。"""
        if current_sim_time >= self.next_log_time:
            current_q = self.mujoco_interface.get_arm_joint_angles()
            torques = self.mujoco_interface.get_joint_torques()
            ee_pos = self.mujoco_interface.get_end_effector_position()

            if current_sim_time % 0.5 < self.config.sim.timestep:
                self.print_robot_info(current_sim_time, ee_pos, current_q)

            self.data_manager.record_data(self.next_log_time, current_q, torques, ee_pos)
            self.next_log_time += self.config.logging.data_logging_interval

    def _execute_simulation_step(self) -> None:
        """读取 SpaceMouse 输入并通过共享 runtime 完成一次控制和仿真步进。"""
        raw_spacemouse_action, buttons = self._read_spacemouse_input()
        if self._handle_spacemouse_reset_request(buttons):
            self._send_joint_velocity_command_to_mujoco(np.zeros_like(self.q_home))
            self.mujoco_interface.step()
            return

        if self.teleop_runtime is None:
            raise RuntimeError("Teleop runtime 尚未初始化。")
        self.teleop_runtime.step(
            raw_spacemouse_action,
            dt=self.config.sim.timestep,
        )

    def _handle_rendering(self, current_sim_time: float) -> None:
        """在仿真循环中处理 viewer 同步。"""
        self._handle_teleop_camera_window(current_sim_time)
        if self.config.visualization.use_viewer:
            self.mujoco_interface.sync_viewer()

    def _setup_teleop_camera_window(self) -> None:
        """初始化末端左右相机实时显示窗口。"""
        visualization = self.config.visualization
        if not visualization.show_teleop_cameras:
            return

        missing_camera_names = [
            camera_name
            for camera_name in visualization.teleop_camera_names
            if not self.mujoco_interface.has_camera(camera_name)
        ]
        if missing_camera_names:
            raise ValueError(
                "当前 MuJoCo 模型缺少实时显示所需相机: "
                f"{missing_camera_names}。请确认使用 velocity 主线模型。"
            )

        self.teleop_camera_window = AsyncCameraWindow(
            model=self.mujoco_interface.model,
            data=self.mujoco_interface.data,
            camera_names=visualization.teleop_camera_names,
            width=visualization.teleop_camera_width,
            height=visualization.teleop_camera_height,
            fps=visualization.teleop_camera_fps,
            window_name=visualization.teleop_camera_window_name,
        )
        logger.info(
            "末端相机窗口已启用: cameras=%s, size=%sx%s, fps=%s",
            visualization.teleop_camera_names,
            visualization.teleop_camera_width,
            visualization.teleop_camera_height,
            visualization.teleop_camera_fps,
        )

    def _handle_teleop_camera_window(self, current_sim_time: float) -> None:
        """提交最新状态并显示后台渲染出的末端相机画面。"""
        del current_sim_time
        if self.teleop_camera_window is None or not self.teleop_camera_window.window_open:
            return

        try:
            self.teleop_camera_window.render()
        except Exception:
            logger.exception("末端相机实时显示失败，已关闭相机窗口。")
            self._close_teleop_camera_window()

    def _close_teleop_camera_window(self) -> None:
        """关闭末端相机 OpenCV 窗口。"""
        if self.teleop_camera_window is None:
            return

        self.teleop_camera_window.close()
        self.teleop_camera_window = None

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

    def _reset_loop_timing(self) -> None:
        """重置所有依赖 MuJoCo 仿真时间的循环调度器。"""
        current_sim_time = self.mujoco_interface.data.time
        self.next_frame_sim_time = current_sim_time
        self.next_log_time = current_sim_time
        self.wall_time_start = time.perf_counter()
        self.sim_time_start = current_sim_time

    def print_robot_info(
        self,
        current_time: float,
        ee_pos: np.ndarray,
        current_q: np.ndarray,
    ) -> None:
        """打印机器人和仿真器的关键信息。"""
        logger.info("=" * 60)
        logger.info("机器人信息:")
        logger.info("仿真时间:%s", current_time)
        logger.info("末端位置:%s", ee_pos)
        logger.info("关节角度: %s", current_q)
        logger.info("=" * 60)

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
                self._reset_loop_timing()
                self._sync_teleop_reference_to_current_state()
            self.reset_button_pressed = True
            return True

        self.reset_button_pressed = reset_pressed
        return False

    def _get_teleop_control_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回共享 teleop runtime 所需的 q、MuJoCo 末端位姿和 LOCAL Jacobian。"""
        current_q = self.mujoco_interface.get_arm_joint_angles()
        current_transform = self.mujoco_interface.get_end_effector_transform()
        jacobian = self.mujoco_interface.get_end_effector_local_jacobian()
        return current_q, current_transform, jacobian

    def _sync_teleop_reference_to_current_state(self) -> None:
        """将 teleop 参考位姿同步到 MuJoCo 当前状态。"""
        self.mujoco_interface.forward()
        if self.teleop_runtime is None:
            raise RuntimeError("Teleop runtime 尚未初始化。")
        self.teleop_runtime.reset_reference()

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
