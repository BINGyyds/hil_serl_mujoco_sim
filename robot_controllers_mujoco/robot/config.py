from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ModeConfig:
    """运行模式配置。"""

    input_mode: str = "planned"             # 输入模式：planned,spacemouse
    control_mode: str = "effort"            # 控制模式：position,effort,velocity
    allow_missing_spacemouse: bool = False


@dataclass
class SimulationConfig:
    """仿真时间与实时同步配置。"""

    simulation_duration: float = 50.0
    timestep: float = 0.001
    sync_to_real_time: bool = True
    real_time_factor: float = 1.0


@dataclass
class VisualizationConfig:
    """可视化与录像配置。"""

    use_viewer: bool = True
    video_fps: int = 30
    video_speed_multiplier: float = 1.0


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
    end_effector_site_name: str = "attachment_site"
    end_effector_pos_sensor_name: str = "attachment_site_pos_sensor"
    attachment_pos_sensor_name: str = "attachment_body_pos_sensor"
    attachment_quat_sensor_name: str = "attachment_body_quat_sensor"
    mjcf_path: Optional[Path] = None


@dataclass
class TeleopConfig:
    """SpaceMouse teleop、位姿伺服与关节安全限制配置。"""

    spacemouse_deadzone: float = 0.05
    spacemouse_filter_alpha: float = 0.25
    max_reference_linear_velocity: float = 0.05
    max_reference_angular_velocity: float = 0.1
    pose_error_gain: float = 25.0
    ik_damping: float = 1e-4
    nullspace_gain: float = 0.15
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


@dataclass
class EffortControlConfig:
    """effort 模式下的关节空间 PD 增益。"""

    kp_gains: Tuple[float, ...] = (
        100.0,
        200.0,
        100.0,
        200.0,
        200.0,
        100.0,
        100.0,
    )
    kd_gains: Tuple[float, ...] = (
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    )


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
    """集中管理 Panda 仿真与后续真机迁移所需的配置。"""

    mode: ModeConfig = field(default_factory=ModeConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    effort_control: EffortControlConfig = field(default_factory=EffortControlConfig)
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
        config.refresh()
        return config

    def refresh(self) -> None:
        """刷新派生字段并校验配置组合是否合法。"""
        self.paths.refresh()
        franka_mjcf_root = self.paths.model_root / "franka_panda"

        if self.mode.input_mode not in {"planned", "spacemouse"}:
            raise ValueError("input_mode 仅支持 'planned' 或 'spacemouse'。")

        if self.mode.control_mode == "position":
            self.model.mjcf_path = franka_mjcf_root / "panda_nohand_position.xml"
        elif self.mode.control_mode == "effort":
            self.model.mjcf_path = franka_mjcf_root / "panda_nohand_effort.xml"
        elif self.mode.control_mode == "velocity":
            self.model.mjcf_path = franka_mjcf_root / "panda_nohand_velocity.xml"
        else:
            raise ValueError(
                "control_mode 仅支持 'position'、'effort' 或 'velocity'。"
            )

        if self.mode.input_mode == "spacemouse" and self.mode.control_mode != "velocity":
            raise ValueError("SpaceMouse teleop 仅支持 control_mode='velocity'。")
        if self.mode.input_mode == "planned" and self.mode.control_mode == "velocity":
            raise ValueError(
                "planned 模式当前仅支持 control_mode='position' 或 'effort'。"
            )
        if self.sim.real_time_factor <= 0.0:
            raise ValueError("real_time_factor 必须大于 0。")
