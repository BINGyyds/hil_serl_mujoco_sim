from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

@dataclass
class ProjectConfig:
    """集中管理 Panda 仿真与后续真机迁移所需的配置。"""

    CONTROL_MODE: str = "effort"      # "position" 或 "effort"，决定使用位置控制还是力控模式

    SIMULATION_DURATION: float = 30.0
    TIMESTEP: float = 0.001

    USE_VIEWER: bool = True
    VIDEO_FPS: int = 30
    VIDEO_SPEED_MULTIPLIER: float = 1.0

    DATA_LOGGING_INTERVAL: float = 0.01

    MUJOCO_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint{i + 1}" for i in range(7))
    MUJOCO_ARM_ACTUATOR_NAMES: Tuple[str, ...] = tuple(f"joint{i + 1}_ctrl" for i in range(7))

    MUJOCO_BASE_SITE_NAME: str = "panda_base_site"
    MUJOCO_END_EFFECTOR_SITE_NAME: str = "attachment_site"

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

        if self.CONTROL_MODE == "position":
            self.MJCF_PATH = franka_mjcf_root / "panda_nohand_position.xml"
        elif self.CONTROL_MODE == "effort":
            self.MJCF_PATH = franka_mjcf_root / "panda_nohand_effort.xml"
        else:
            raise ValueError("CONTROL_MODE 仅支持 'position' 或 'effort'。")

        self.MAIN_VIDEO_PATH = self.output_root / "videos" / "main_simulation.mp4"
        self.ROBOT_DATA_PATH = self.output_root / "docs" / "robot_data.csv"
