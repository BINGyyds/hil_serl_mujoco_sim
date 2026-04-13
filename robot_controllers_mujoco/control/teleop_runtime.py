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
    """Result from one shared teleop/action control step."""

    joint_velocity_command: np.ndarray


class TeleopRuntime:
    """
    Shared task-space velocity runtime.

    Action convention:
        * action[0:3] is a world-frame translation command.
        * action[3:6] is an end-effector-local rotation command.

    The frame semantics are implemented by TaskSpaceVelocityController; this
    class owns the surrounding runtime plumbing so manual teleop, demo
    recording, and actor intervention call the same control path.
    """

    def __init__(
        self,
        *,
        controller: TaskSpaceVelocityController,
        get_control_state: ControlStateProvider,
        set_joint_velocity: JointVelocitySetter,
        step_simulation: SimulationStepper,
        control_dt: float,
        get_wrench_world: Optional[WrenchProvider] = None,
        tare_wrench: Optional[WrenchTare] = None,
        enable_admittance: bool = False,
    ) -> None:
        self.controller = controller
        self.get_control_state = get_control_state
        self.set_joint_velocity = set_joint_velocity
        self.step_simulation = step_simulation
        self.control_dt = float(control_dt)
        self.get_wrench_world = get_wrench_world
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
        wrench_world = None
        if self.enable_admittance and self.get_wrench_world is not None:
            wrench_world = self.get_wrench_world()

        return self.controller.compute_command(
            raw_action=np.asarray(action, dtype=np.float64).reshape((6,)),
            current_q=current_q,
            current_transform=current_transform,
            jacobian=jacobian,
            dt=self.control_dt if dt is None else float(dt),
            wrench_world=wrench_world,
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
