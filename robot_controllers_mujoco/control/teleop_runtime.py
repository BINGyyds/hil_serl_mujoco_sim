from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from robot_controllers_mujoco.control.task_space_velocity_controller import (
    TaskSpaceVelocityController,
)


ControlStateProvider = Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray]]
JointVelocitySetter = Callable[[np.ndarray], None]
SimulationStepper = Callable[[], None]
WrenchProvider = Callable[[], np.ndarray]
WrenchTare = Callable[[], None]


@dataclass(frozen=True)
class TeleopStepResult:
    """一次共享 teleop/action 控制步的结果。"""

    joint_velocity_command: np.ndarray


class TeleopRuntime:
    """共享任务空间速度控制运行时。

    action 约定：
        * action[0:3] 是世界系平移命令。
        * action[3:6] 是末端本体系旋转命令。

    坐标系语义由 TaskSpaceVelocityController 实现；该类只负责外围运行时胶水，
    使手动 teleop、demo 录制和 actor intervention 可以调用同一条控制路径。
    """

    def __init__(
        self,
        *,
        controller: TaskSpaceVelocityController,
        get_control_state: ControlStateProvider,
        set_joint_velocity: JointVelocitySetter,
        step_simulation: SimulationStepper,
        control_dt: float,
        get_wrench_local: Optional[WrenchProvider] = None,
        tare_wrench: Optional[WrenchTare] = None,
        enable_admittance: bool = False,
    ) -> None:
        self.controller = controller
        self.get_control_state = get_control_state
        self.set_joint_velocity = set_joint_velocity
        self.step_simulation = step_simulation
        self.control_dt = float(control_dt)
        self.get_wrench_local = get_wrench_local
        self.tare_wrench = tare_wrench
        self.enable_admittance = bool(enable_admittance)

    def reset_reference(self) -> None:
        current_q, current_transform, _ = self.get_control_state()
        self.controller.reset(
            current_transform=current_transform,
            current_q=current_q,
        )
        if self.enable_admittance and self.tare_wrench is not None:
            self.tare_wrench()

    def compute_joint_velocity(
        self,
        action: np.ndarray,
        *,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        current_q, current_transform, jacobian = self.get_control_state()
        wrench_local = None
        if self.enable_admittance and self.get_wrench_local is not None:
            wrench_local = self.get_wrench_local()

        return self.controller.compute_command(
            raw_action=np.asarray(action, dtype=np.float64).reshape((6,)),
            current_q=current_q,
            current_transform=current_transform,
            jacobian=jacobian,
            dt=self.control_dt if dt is None else float(dt),
            wrench_local=wrench_local,
        )

    def step(
        self,
        action: np.ndarray,
        *,
        dt: Optional[float] = None,
    ) -> TeleopStepResult:
        joint_velocity_command = self.compute_joint_velocity(action, dt=dt)
        self.set_joint_velocity(joint_velocity_command)
        self.step_simulation()
        return TeleopStepResult(joint_velocity_command=joint_velocity_command)
