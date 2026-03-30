from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

@dataclass
class ProjectConfig:
    """集中管理 Panda 仿真与后续真机迁移所需的配置。"""

    INPUT_MODE: str = "planned"       # "planned" 或 "spacemouse"
    CONTROL_MODE: str = "effort"      # "position"、"effort" 或 "velocity"
    ALLOW_MISSING_SPACEMOUSE: bool = False

    SIMULATION_DURATION: float = 50.0
    TIMESTEP: float = 0.002

    USE_VIEWER: bool = True
    VIDEO_FPS: int = 30
    VIDEO_SPEED_MULTIPLIER: float = 1.0

    DATA_LOGGING_INTERVAL: float = 0.01

    MUJOCO_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint{i + 1}" for i in range(7))
    MUJOCO_ARM_ACTUATOR_NAMES: Tuple[str, ...] = tuple(f"joint{i + 1}_ctrl" for i in range(7))
    MUJOCO_ARM_JOINT_POS_SENSOR_NAMES: Tuple[str, ...] = tuple(
        f"joint{i + 1}_pos_sensor" for i in range(7)
    )
    MUJOCO_ARM_JOINT_VEL_SENSOR_NAMES: Tuple[str, ...] = tuple(
        f"joint{i + 1}_vel_sensor" for i in range(7)
    )
    MUJOCO_ARM_JOINT_TORQUE_SENSOR_NAMES: Tuple[str, ...] = tuple(
        f"joint{i + 1}_torque_sensor" for i in range(7)
    )

    MUJOCO_BASE_SITE_NAME: str = "panda_base_site"
    MUJOCO_END_EFFECTOR_SITE_NAME: str = "attachment_site"
    MUJOCO_END_EFFECTOR_POS_SENSOR_NAME: str = "attachment_site_pos_sensor"
    MUJOCO_ATTACHMENT_POS_SENSOR_NAME: str = "attachment_body_pos_sensor"
    MUJOCO_ATTACHMENT_QUAT_SENSOR_NAME: str = "attachment_body_quat_sensor"

    SPACEMOUSE_DEADZONE: float = 0.05
    SPACEMOUSE_TRANSLATION_SCALE: float = 0.20
    SPACEMOUSE_ROTATION_SCALE: float = 0.80
    SPACEMOUSE_FILTER_ALPHA: float = 0.25
    TELEOP_POSITION_KP: float = 8.0
    TELEOP_ROTATION_KP: float = 6.0
    JOINT_VELOCITY_LIMITS: Tuple[float, ...] = (
        2.175,
        2.175,
        2.175,
        2.175,
        2.610,
        2.610,
        2.610,
    )
    JOINT_LIMIT_MARGIN: float = 0.05
    IK_DAMPING: float = 1e-4
    NULLSPACE_GAIN: float = 0.15

    # 力控模式 (effort) 下的PD反馈控制器增益
    EFFORT_KP_GAINS: Tuple[float, ...] = (100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 100.0) # 比例增益 (Kp)
    EFFORT_KD_GAINS: Tuple[float, ...] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)      # 微分增益 (Kd)

    package_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1],
        init=False,
    )
    repo_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        init=False,
    )
    model_root: Path = field(init=False)
    output_root: Path = field(init=False)

    MJCF_PATH: Path = field(init=False)

    MAIN_VIDEO_PATH: Path = field(init=False)
    ROBOT_DATA_PATH: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_root = self.package_root / "models"
        self.output_root = self.repo_root / "output"
        franka_mjcf_root = self.model_root / "franka_panda"

        if self.INPUT_MODE not in {"planned", "spacemouse"}:
            raise ValueError("INPUT_MODE 仅支持 'planned' 或 'spacemouse'。")

        if self.CONTROL_MODE == "position":
            self.MJCF_PATH = franka_mjcf_root / "panda_nohand_position.xml"
        elif self.CONTROL_MODE == "effort":
            self.MJCF_PATH = franka_mjcf_root / "panda_nohand_effort.xml"
        elif self.CONTROL_MODE == "velocity":
            self.MJCF_PATH = franka_mjcf_root / "panda_nohand_velocity.xml"
        else:
            raise ValueError("CONTROL_MODE 仅支持 'position'、'effort' 或 'velocity'。")

        if self.INPUT_MODE == "spacemouse" and self.CONTROL_MODE != "velocity":
            raise ValueError(
                "SpaceMouse teleop 仅支持 CONTROL_MODE='velocity'。"
            )
        if self.INPUT_MODE == "planned" and self.CONTROL_MODE == "velocity":
            raise ValueError(
                "planned 模式当前仅支持 CONTROL_MODE='position' 或 'effort'。"
            )

        self.MAIN_VIDEO_PATH = self.output_root / "videos" / "main_simulation.mp4"
        self.ROBOT_DATA_PATH = self.output_root / "docs" / "robot_data.csv"
