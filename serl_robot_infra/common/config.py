from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ModeConfig:
    """运行模式配置。"""

    input_mode: str = "spacemouse"
    control_mode: str = "torque"
    allow_missing_spacemouse: bool = False


@dataclass
class SimulationConfig:
    """仿真时间与实时同步配置。"""

    simulation_duration: float = 10.0
    timestep: float = 0.001
    debug_control_dt: float = 0.01
    enable_gravity_compensation: bool = False
    sync_to_real_time: bool = True
    real_time_factor: float = 1.0


@dataclass
class VisualizationConfig:
    """可视化与录像配置。"""

    use_viewer: bool = True
    video_fps: int = 30
    video_speed_multiplier: float = 1.0
    show_teleop_cameras: bool = False
    teleop_camera_names: Tuple[str, ...] = (
        "wrist_left_camera",
        "wrist_right_camera",
    )
    teleop_camera_fps: int = 20
    teleop_camera_width: int = 320
    teleop_camera_height: int = 240
    teleop_camera_window_name: str = "wrist cameras"


@dataclass
class LoggingConfig:
    """数据采样与控制台输出配置。"""

    data_logging_interval: float = 0.01


@dataclass
class ModelConfig:
    """MuJoCo 模型命名、传感器与派生路径配置。"""

    arm_joint_names: Tuple[str, ...] = tuple(f"joint{i + 1}" for i in range(7))
    arm_actuator_names: Tuple[str, ...] = tuple(f"joint{i + 1}_ctrl" for i in range(7))
    arm_joint_pos_sensor_names: Tuple[str, ...] = tuple(
        f"joint{i + 1}_pos_sensor" for i in range(7)
    )
    arm_joint_vel_sensor_names: Tuple[str, ...] = tuple(
        f"joint{i + 1}_vel_sensor" for i in range(7)
    )
    arm_joint_torque_sensor_names: Tuple[str, ...] = tuple(
        f"joint{i + 1}_torque_sensor" for i in range(7)
    )
    base_site_name: str = "panda_base_site"
    end_effector_site_name: str = "tool_tip_site"
    control_frame_site_name: str = "tool_mount"
    end_effector_pos_sensor_name: Optional[str] = None
    end_effector_force_sensor_name: Optional[str] = None
    end_effector_torque_sensor_name: Optional[str] = None
    attachment_pos_sensor_name: str = "attachment_body_pos_sensor"
    attachment_quat_sensor_name: str = "attachment_body_quat_sensor"
    mjcf_path: Optional[Path] = None
    nominal_mjcf_path: Optional[Path] = None


@dataclass
class TeleopConfig:
    """SpaceMouse teleop、位姿伺服与关节安全限制配置。"""

    spacemouse_deadzone: float = 0.05
    spacemouse_filter_alpha: float = 0.25
    max_reference_linear_velocity: float = 0.05
    max_reference_angular_velocity: float = 0.1
    max_reference_linear_offset: float = 0.01
    max_reference_angular_offset: float = 0.2
    ik_damping: float = 1e-4
    joint_velocity_limits: Tuple[float, ...] = (
        2.175,
        2.175,
        2.175,
        2.175,
        2.610,
        2.610,
        2.610,
    )
    joint_limit_margin: float = 0.05
    control_frame_name: str = "attachment"
    use_external_wrench_feedback: bool = False
    enable_nric: bool = True
    impedance_inertia: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.1, 0.1, 0.1)
    impedance_stiffness: Tuple[float, ...] = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0)
    impedance_damping: Tuple[float, ...] = (
        10.0,
        10.0,
        10.0,
        1.0,
        1.0,
        1.0,
    )
    nullspace_stiffness: float = 10.0
    nullspace_damping: float = 2.0
    nric_k_gamma: Tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    nric_k_p: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    nric_k_i: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    nric_integral_limit: float = 0.2
    nric_integral_deadband: float = 5e-3
    nric_integral_leak_rate: float = 0.2
    nric_zero_cross_reset_ratio: float = 0.2


@dataclass
class PathsConfig:
    """工程路径与输出文件路径配置。"""

    package_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    repo_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    model_root: Path = field(init=False)
    output_root: Path = field(init=False)
    main_video_path: Path = field(init=False)
    robot_data_path: Path = field(init=False)

    def refresh(self) -> None:
        """根据项目根目录刷新派生路径。"""
        self.model_root = self.package_root / "models"
        self.output_root = self.repo_root / "output"
        self.main_video_path = self.output_root / "videos" / "main_simulation.mp4"
        self.robot_data_path = self.output_root / "docs" / "robot_data.csv"


@dataclass
class ProjectConfig:
    """集中管理当前 MuJoCo teleop/debug 主线所需的配置。"""

    mode: ModeConfig = field(default_factory=ModeConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self) -> None:
        self.refresh()

    @classmethod
    def from_overrides(
        cls,
        *,
        use_viewer: Optional[bool] = None,
        input_mode: Optional[str] = None,
        control_mode: Optional[str] = None,
        allow_missing_spacemouse: Optional[bool] = None,
        show_teleop_cameras: Optional[bool] = None,
        teleop_camera_fps: Optional[int] = None,
        teleop_camera_width: Optional[int] = None,
        teleop_camera_height: Optional[int] = None,
    ) -> "ProjectConfig":
        """使用默认配置并应用少量运行时覆盖项。"""
        config = cls()
        if use_viewer is not None:
            config.visualization.use_viewer = use_viewer
        if input_mode is not None:
            config.mode.input_mode = input_mode
        if control_mode is not None:
            config.mode.control_mode = control_mode
        if allow_missing_spacemouse is not None:
            config.mode.allow_missing_spacemouse = allow_missing_spacemouse
        if show_teleop_cameras is not None:
            config.visualization.show_teleop_cameras = show_teleop_cameras
        if teleop_camera_fps is not None:
            config.visualization.teleop_camera_fps = teleop_camera_fps
        if teleop_camera_width is not None:
            config.visualization.teleop_camera_width = teleop_camera_width
        if teleop_camera_height is not None:
            config.visualization.teleop_camera_height = teleop_camera_height
        config.refresh()
        return config

    def refresh(self) -> None:
        """刷新派生字段并校验当前唯一支持的运行组合。"""
        self.paths.refresh()
        franka_mjcf_root = self.paths.model_root / "franka_panda"

        if self.mode.input_mode != "spacemouse":
            raise ValueError("当前 debug/teleop 主线仅支持 input_mode='spacemouse'。")
        if self.mode.control_mode != "torque":
            raise ValueError(
                "当前 debug/teleop 主线仅支持 control_mode='torque'。"
            )

        self.model.mjcf_path = franka_mjcf_root / "panda_nohand_effort.xml"
        self.model.nominal_mjcf_path = franka_mjcf_root / "panda_nohand.xml"
        self.model.end_effector_force_sensor_name = "attachment_site_force_sensor"
        self.model.end_effector_torque_sensor_name = "attachment_site_torque_sensor"
        self.sim.enable_gravity_compensation = False

        if self.sim.real_time_factor <= 0.0:
            raise ValueError("real_time_factor 必须大于 0。")
        if self.sim.debug_control_dt <= 0.0:
            raise ValueError("debug_control_dt 必须大于 0。")
        if self.sim.debug_control_dt < self.sim.timestep:
            raise ValueError("debug_control_dt 不能小于仿真 timestep。")
        control_ratio = self.sim.debug_control_dt / self.sim.timestep
        if not abs(control_ratio - round(control_ratio)) < 1e-9:
            raise ValueError(
                "debug_control_dt 必须是 timestep 的整数倍，"
                f"当前比值为 {control_ratio:.6f}。"
            )
        if self.visualization.teleop_camera_fps <= 0:
            raise ValueError("teleop_camera_fps 必须大于 0。")
        if self.visualization.teleop_camera_width <= 0:
            raise ValueError("teleop_camera_width 必须大于 0。")
        if self.visualization.teleop_camera_height <= 0:
            raise ValueError("teleop_camera_height 必须大于 0。")
