"""Shared torque-control runtime for teleop, demo recording, and RL env steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from serl_robot_infra.common.se3_utils import (
    compose_transforms,
    exp_se3,
    invert_transform,
    log_se3,
)
from serl_robot_infra.control.nric_torque_controller import (
    NRICTorqueController,
    NRICTorqueResult,
)
from serl_robot_infra.control.se3_torque_controller import SE3TorqueController


TorqueControlStateProvider = Callable[[], dict[str, np.ndarray]]
JointTorqueSetter = Callable[[np.ndarray], None]
SimulationStepper = Callable[[], None]


@dataclass(frozen=True)
class TorqueTeleopStepResult:
    joint_torque_command: np.ndarray
    nominal_torque: np.ndarray
    auxiliary_torque: np.ndarray


class SE3TorqueRuntime:
    """Glue runtime that keeps the high-level 6D action API unchanged."""

    def __init__(
        self,
        *,
        controller: SE3TorqueController,
        nric_controller: NRICTorqueController,
        get_control_state: TorqueControlStateProvider,
        set_joint_torque: JointTorqueSetter,
        step_simulation: SimulationStepper,
        control_dt: float,
        sim_steps_per_control: int = 1,
        tip_to_control_transform: np.ndarray,
        input_deadzone: float,
        input_filter_alpha: float,
        max_reference_linear_velocity: float,
        max_reference_angular_velocity: float,
        max_reference_linear_offset: float,
        max_reference_angular_offset: float,
        use_external_wrench_feedback: bool = False,
    ) -> None:
        self.controller = controller
        self.nric_controller = nric_controller
        self.get_control_state = get_control_state
        self.set_joint_torque = set_joint_torque
        self.step_simulation = step_simulation
        self.control_dt = float(control_dt)
        self.sim_steps_per_control = max(1, int(sim_steps_per_control))
        self.tip_to_control_transform = np.asarray(
            tip_to_control_transform,
            dtype=np.float64,
        ).reshape((4, 4))
        self.input_deadzone = float(input_deadzone)
        self.input_filter_alpha = float(np.clip(input_filter_alpha, 0.0, 1.0))
        self.max_reference_linear_velocity = float(max_reference_linear_velocity)
        self.max_reference_angular_velocity = float(max_reference_angular_velocity)
        self.max_reference_linear_offset = max(0.0, float(max_reference_linear_offset))
        self.max_reference_angular_offset = max(
            0.0,
            float(max_reference_angular_offset),
        )
        self.use_external_wrench_feedback = bool(use_external_wrench_feedback)
        self.filtered_command = np.zeros(6, dtype=np.float64)
        self.target_tip_transform: Optional[np.ndarray] = None

    def reset_reference(self) -> None:
        state = self.get_control_state()
        self.target_tip_transform = np.asarray(
            state["tip_transform"],
            dtype=np.float64,
        ).copy()
        self.filtered_command[:] = 0.0
        self.nric_controller.reset(state["q"], state["q_dot"])

    def get_target_tip_transform(self) -> Optional[np.ndarray]:
        if self.target_tip_transform is None:
            return None
        return self.target_tip_transform.copy()

    def get_internal_state(self) -> dict[str, object]:
        return {
            "filtered_command": self.filtered_command.copy(),
            "target_tip_transform": None
            if self.target_tip_transform is None
            else self.target_tip_transform.copy(),
            "nric_state": self.nric_controller.get_internal_state(),
        }

    def set_internal_state(self, state: dict[str, object]) -> None:
        self.filtered_command = np.asarray(
            state["filtered_command"],
            dtype=np.float64,
        ).copy()
        self.target_tip_transform = None
        if state.get("target_tip_transform") is not None:
            self.target_tip_transform = np.asarray(
                state["target_tip_transform"],
                dtype=np.float64,
            ).copy()
        self.nric_controller.set_internal_state(state["nric_state"])  # type: ignore[arg-type]

    def compute_joint_torque(
        self,
        action: np.ndarray,
        *,
        dt: Optional[float] = None,
    ) -> NRICTorqueResult:
        state = self.get_control_state()
        if self.target_tip_transform is None:
            self.reset_reference()
            state = self.get_control_state()

        filtered_command = self._filter_input(np.asarray(action, dtype=np.float64).reshape((6,)))
        self._integrate_target_tip_transform(filtered_command, dt=self.control_dt if dt is None else float(dt))
        self._project_target_tip_transform(state["tip_transform"])
        desired_control_transform = compose_transforms(
            self.target_tip_transform,
            self.tip_to_control_transform,
        )

        nominal_result = self.controller.compute_torque(
            q=state["q"],
            q_dot=state["q_dot"],
            current_transform=state["control_transform"],
            current_twist_body=state["control_twist_body"],
            jacobian_body=state["control_jacobian_body"],
            mass_matrix=state["mass_matrix"],
            bias_forces=state["bias_forces"],
            desired_transform=desired_control_transform,
            desired_twist_body=np.zeros(6, dtype=np.float64),
            desired_twist_dot_body=np.zeros(6, dtype=np.float64),
            external_wrench_body=(
                state["wrench_body"]
                if self.use_external_wrench_feedback
                else np.zeros(6, dtype=np.float64)
            ),
        )
        return self.nric_controller.compute_torque(
            q=state["q"],
            q_dot=state["q_dot"],
            nominal_torque_mujoco=nominal_result.joint_torque,
            desired_transform=desired_control_transform,
            desired_twist_body=np.zeros(6, dtype=np.float64),
            desired_twist_dot_body=np.zeros(6, dtype=np.float64),
            external_wrench_body=(
                state["wrench_body"]
                if self.use_external_wrench_feedback
                else np.zeros(6, dtype=np.float64)
            ),
        )

    def step(
        self,
        action: np.ndarray,
        *,
        dt: Optional[float] = None,
    ) -> TorqueTeleopStepResult:
        torque_result = self.compute_joint_torque(action, dt=dt)
        self.set_joint_torque(torque_result.joint_torque)
        for _ in range(self.sim_steps_per_control):
            self.step_simulation()
        return TorqueTeleopStepResult(
            joint_torque_command=torque_result.joint_torque.copy(),
            nominal_torque=torque_result.nominal_torque_mujoco.copy(),
            auxiliary_torque=torque_result.auxiliary_torque.copy(),
        )

    def _filter_input(self, raw_action: np.ndarray) -> np.ndarray:
        action = raw_action.copy()
        action[np.abs(action) < self.input_deadzone] = 0.0
        if not np.any(action):
            self.filtered_command[:] = 0.0
            return self.filtered_command.copy()
        self.filtered_command = (
            self.input_filter_alpha * action
            + (1.0 - self.input_filter_alpha) * self.filtered_command
        )
        return self.filtered_command.copy()

    def _integrate_target_tip_transform(
        self,
        filtered_command: np.ndarray,
        *,
        dt: float,
    ) -> None:
        if self.target_tip_transform is None:
            raise RuntimeError("Target tip transform is not initialized.")
        body_twist = np.zeros(6, dtype=np.float64)
        body_twist[:3] = filtered_command[:3] * dt * self.max_reference_linear_velocity
        body_twist[3:] = filtered_command[3:] * dt * self.max_reference_angular_velocity
        self.target_tip_transform = compose_transforms(
            self.target_tip_transform,
            exp_se3(body_twist),
        )

    def _project_target_tip_transform(self, current_tip_transform: np.ndarray) -> None:
        if self.target_tip_transform is None:
            raise RuntimeError("Target tip transform is not initialized.")

        relative_error = log_se3(
            compose_transforms(
                invert_transform(current_tip_transform),
                self.target_tip_transform,
            )
        )
        linear_error = relative_error[:3]
        angular_error = relative_error[3:]

        linear_norm = np.linalg.norm(linear_error)
        if self.max_reference_linear_offset > 0.0 and linear_norm > self.max_reference_linear_offset:
            linear_error = linear_error * (self.max_reference_linear_offset / linear_norm)

        angular_norm = np.linalg.norm(angular_error)
        if self.max_reference_angular_offset > 0.0 and angular_norm > self.max_reference_angular_offset:
            angular_error = angular_error * (self.max_reference_angular_offset / angular_norm)

        clipped_relative_error = np.concatenate([linear_error, angular_error])
        self.target_tip_transform = compose_transforms(
            current_tip_transform,
            exp_se3(clipped_relative_error),
        )
