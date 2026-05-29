import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from serl_robot_infra.common.async_camera_window import AsyncCameraWindow
from serl_robot_infra.common.async_writer import AsyncVideoWriter
from serl_robot_infra.common.config import ProjectConfig
from serl_robot_infra.common.data_manager import DataManager
from serl_robot_infra.common.se3_utils import compose_transforms, invert_transform
from serl_robot_infra.control.nric_torque_controller import NRICTorqueController
from serl_robot_infra.control.se3_torque_controller import SE3TorqueController
from serl_robot_infra.control.se3_torque_runtime import SE3TorqueRuntime
from serl_robot_infra.simulation.mujoco_interface import MujocoInterface

logger = logging.getLogger(__name__)


class RobotController:
    """Franka insertion 的本地 SpaceMouse + torque teleop 控制器。"""

    def __init__(self, config: ProjectConfig) -> None:
        logger.info("正在初始化机器人控制器...")
        self.config = config
        self.mujoco_interface = MujocoInterface(config)
        self.data_manager = DataManager()

        self.spacemouse: Optional[Any] = None
        self.teleop_runtime: Optional[SE3TorqueRuntime] = None
        self.video_writers: List[AsyncVideoWriter] = []
        self.teleop_camera_window: Optional[AsyncCameraWindow] = None

        self.reset_button_pressed = False
        self.reanchor_button_pressed = False
        self.q_home = np.zeros(7, dtype=np.float64)
        self.tip_to_control_transform = np.eye(4, dtype=np.float64)

        self.wall_time_start = 0.0
        self.sim_time_start = 0.0
        self.video_frame_duration = 0.0
        self.next_frame_sim_time = 0.0
        self.next_log_time = 0.0

        self.task_space_controller = SE3TorqueController(
            apparent_inertia=np.asarray(config.teleop.impedance_inertia, dtype=np.float64),
            lambda_stiffness=np.asarray(config.teleop.impedance_stiffness, dtype=np.float64),
            physical_damping=np.asarray(config.teleop.impedance_damping, dtype=np.float64),
            q_nominal=None,
            nullspace_stiffness=config.teleop.nullspace_stiffness,
            nullspace_damping=config.teleop.nullspace_damping,
        )
        self.nric_controller = NRICTorqueController(
            nominal_mjcf_path=str(config.model.nominal_mjcf_path),
            end_effector_frame_name=config.teleop.control_frame_name,
            time_step=config.sim.debug_control_dt,
            gravity_z=float(self.mujoco_interface.model.opt.gravity[2]),
            enabled=config.teleop.enable_nric,
            q_nominal=None,
            apparent_inertia=np.asarray(config.teleop.impedance_inertia, dtype=np.float64),
            lambda_stiffness=np.asarray(config.teleop.impedance_stiffness, dtype=np.float64),
            physical_damping=np.asarray(config.teleop.impedance_damping, dtype=np.float64),
            nullspace_stiffness=config.teleop.nullspace_stiffness,
            nullspace_damping=config.teleop.nullspace_damping,
            k_gamma=np.asarray(config.teleop.nric_k_gamma, dtype=np.float64),
            k_p=np.asarray(config.teleop.nric_k_p, dtype=np.float64),
            k_i=np.asarray(config.teleop.nric_k_i, dtype=np.float64),
            integral_limit=config.teleop.nric_integral_limit,
            integral_deadband=config.teleop.nric_integral_deadband,
            integral_leak_rate=config.teleop.nric_integral_leak_rate,
            zero_cross_reset_ratio=config.teleop.nric_zero_cross_reset_ratio,
        )
        self.teleop_runtime = SE3TorqueRuntime(
            controller=self.task_space_controller,
            nric_controller=self.nric_controller,
            get_control_state=self._get_teleop_control_state,
            set_joint_torque=self._send_joint_torque_command_to_mujoco,
            step_simulation=self.mujoco_interface.step,
            control_dt=config.sim.debug_control_dt,
            sim_steps_per_control=max(
                1,
                int(round(config.sim.debug_control_dt / config.sim.timestep)),
            ),
            tip_to_control_transform=self.tip_to_control_transform,
            input_deadzone=config.teleop.spacemouse_deadzone,
            input_filter_alpha=config.teleop.spacemouse_filter_alpha,
            max_reference_linear_velocity=config.teleop.max_reference_linear_velocity,
            max_reference_angular_velocity=config.teleop.max_reference_angular_velocity,
            max_reference_linear_offset=config.teleop.max_reference_linear_offset,
            max_reference_angular_offset=config.teleop.max_reference_angular_offset,
            use_external_wrench_feedback=config.teleop.use_external_wrench_feedback,
            # In manual controller debugging, keep the compliant equilibrium at
            # the reset/re-anchor pose instead of letting contact motion drag the
            # reference along with the current end-effector pose.
            project_reference_to_current=False,
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
        self.tip_to_control_transform = compose_transforms(
            invert_transform(self.mujoco_interface.get_end_effector_transform()),
            self.mujoco_interface.get_control_frame_transform(),
        )
        if self.teleop_runtime is not None:
            self.teleop_runtime.tip_to_control_transform = (
                self.tip_to_control_transform.copy()
            )
        self._reset_loop_timing()
        self.q_home = self.mujoco_interface.get_arm_joint_angles()
        self.task_space_controller.q_nominal = self.q_home.copy()

        self._initialize_spacemouse_mode()
        self.mujoco_interface.tare_end_effector_wrench()

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
        self.reanchor_button_pressed = False
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
        self._handle_spacemouse_reanchor_request(buttons)
        if self._handle_spacemouse_reset_request(buttons):
            if self.teleop_runtime is None:
                raise RuntimeError("Teleop runtime 尚未初始化。")
            self.teleop_runtime.step(
                np.zeros(6, dtype=np.float64),
                dt=self.config.sim.debug_control_dt,
            )
            return

        if self.teleop_runtime is None:
            raise RuntimeError("Teleop runtime 尚未初始化。")
        self.teleop_runtime.step(
            raw_spacemouse_action,
            dt=self.config.sim.debug_control_dt,
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
                f"{missing_camera_names}。请确认使用 insertion 主线模型。"
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

    def _handle_spacemouse_reanchor_request(self, buttons: list) -> None:
        """用右键将 teleop 目标重新锚定到当前真实末端位姿。"""
        reanchor_pressed = bool(buttons[1]) if len(buttons) > 1 else False
        if reanchor_pressed and not self.reanchor_button_pressed:
            self._sync_teleop_reference_to_current_state()
        self.reanchor_button_pressed = reanchor_pressed

    def _get_teleop_control_state(self) -> Dict[str, np.ndarray]:
        """返回共享 torque runtime 所需的机器人状态。"""
        return {
            "q": self.mujoco_interface.get_arm_joint_angles(),
            "q_dot": self.mujoco_interface.get_arm_joint_velocities(),
            "tip_transform": self.mujoco_interface.get_end_effector_transform(),
            "control_transform": self.mujoco_interface.get_control_frame_transform(),
            "control_jacobian_body": self.mujoco_interface.get_control_frame_local_jacobian(),
            "control_twist_body": self.mujoco_interface.get_control_frame_local_twist(),
            "mass_matrix": self.mujoco_interface.get_arm_mass_matrix(),
            "bias_forces": self.mujoco_interface.get_arm_bias_forces(),
            "wrench_body": self._get_control_frame_wrench_body(),
        }

    def _sync_teleop_reference_to_current_state(self) -> None:
        """将 teleop 参考位姿同步到 MuJoCo 当前状态。"""
        self.mujoco_interface.forward()
        self.mujoco_interface.tare_end_effector_wrench()
        if self.teleop_runtime is None:
            raise RuntimeError("Teleop runtime 尚未初始化。")
        self.teleop_runtime.reset_reference()

    def _send_joint_torque_command_to_mujoco(
        self,
        joint_torque_command: np.ndarray,
    ) -> None:
        """将关节力矩命令下发到 MuJoCo。"""
        joint_torque = np.asarray(joint_torque_command, dtype=np.float64)
        expected_shape = (len(self.config.model.arm_joint_names),)
        if joint_torque.shape != expected_shape:
            raise ValueError(
                "关节力矩命令维度错误，"
                f"期望 {expected_shape}，"
                f"实际 {joint_torque.shape}"
            )
        self.mujoco_interface.set_control(joint_torque)

    def _get_control_frame_wrench_body(self) -> np.ndarray:
        return self.mujoco_interface.get_end_effector_wrench_local()
