"""Pinocchio-backed NRIC wrapper for the insertion torque controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

from serl_robot_infra.common.se3_gic import SE3GeometryImpedance
from serl_robot_infra.control.se3_torque_controller import (
    SE3TorqueControlResult,
    SE3TorqueController,
)

NV_ARM = 7


def _as_vector7(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (NV_ARM,):
        raise ValueError(f"{label} must have shape ({NV_ARM},), got {array.shape}")
    return array


def _as_diag7(value: Optional[np.ndarray], *, default: float) -> np.ndarray:
    if value is None:
        return np.eye(NV_ARM, dtype=np.float64) * default
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (NV_ARM,):
        return np.diag(array)
    if array.shape == (NV_ARM, NV_ARM):
        return array.copy()
    raise ValueError(f"Expected shape ({NV_ARM},) or ({NV_ARM}, {NV_ARM}), got {array.shape}")


def _build_nominal_pinocchio_model(
    *,
    mjcf_path: str,
    gravity_z: float,
) -> pin.Model:
    model_path = Path(mjcf_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Pinocchio nominal MJCF not found: {mjcf_path}")
    model = pin.buildModelFromMJCF(str(model_path))
    model.gravity.linear[:] = 0.0
    model.gravity.angular[:] = 0.0
    model.gravity.linear[2] = float(gravity_z)
    if model.nq != NV_ARM or model.nv != NV_ARM:
        raise RuntimeError(
            f"Expected a fixed-base 7-DoF Panda nominal model, got nq={model.nq}, nv={model.nv}."
        )
    return model


def _lookup_frame_id(model: pin.Model, preferred_name: str) -> int:
    if model.existFrame(preferred_name):
        return int(model.getFrameId(preferred_name))
    fallback_names = ["tool_mount", "attachment_site", "attachment", "panda_link8_tcp"]
    for fallback in fallback_names:
        if model.existFrame(fallback):
            return int(model.getFrameId(fallback))
    raise RuntimeError(
        f"End-effector frame '{preferred_name}' not found in Pinocchio model and no fallback frame is available."
    )


class PinocchioNominalSE3Controller:
    """Mirror SE(3) torque control on the nominal Pinocchio model."""

    def __init__(
        self,
        *,
        mjcf_path: str,
        end_effector_frame_name: str,
        gravity_z: float,
        gic: Optional[SE3GeometryImpedance] = None,
        q_nominal: Optional[np.ndarray] = None,
        apparent_inertia: Optional[np.ndarray] = None,
        lambda_stiffness: Optional[np.ndarray] = None,
        physical_damping: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
    ) -> None:
        self.model = _build_nominal_pinocchio_model(
            mjcf_path=mjcf_path,
            gravity_z=gravity_z,
        )
        self.data = self.model.createData()
        self.frame_id = _lookup_frame_id(self.model, end_effector_frame_name)
        self.controller = SE3TorqueController(
            gic=gic,
            apparent_inertia=apparent_inertia,
            lambda_stiffness=lambda_stiffness,
            physical_damping=physical_damping,
            q_nominal=q_nominal,
            nullspace_stiffness=nullspace_stiffness,
            nullspace_damping=nullspace_damping,
        )

    def compute_torque(
        self,
        *,
        q: np.ndarray,
        q_dot: np.ndarray,
        desired_transform: np.ndarray,
        desired_twist_body: Optional[np.ndarray] = None,
        desired_twist_dot_body: Optional[np.ndarray] = None,
        external_wrench_body: Optional[np.ndarray] = None,
    ) -> SE3TorqueControlResult:
        q = _as_vector7(q, "q")
        q_dot = _as_vector7(q_dot, "q_dot")
        jacobian = np.asarray(
            pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                pin.ReferenceFrame.LOCAL,
            ),
            dtype=np.float64,
        )
        pin.forwardKinematics(self.model, self.data, q, q_dot, np.zeros(NV_ARM, dtype=np.float64))
        pin.updateFramePlacements(self.model, self.data)

        current_transform = np.asarray(self.data.oMf[self.frame_id].homogeneous, dtype=np.float64)
        current_twist = np.asarray(
            pin.getFrameVelocity(
                self.model,
                self.data,
                self.frame_id,
                pin.ReferenceFrame.LOCAL,
            ).vector,
            dtype=np.float64,
        )
        bias_forces = np.asarray(pin.nonLinearEffects(self.model, self.data, q, q_dot), dtype=np.float64)
        mass_matrix = np.asarray(pin.crba(self.model, self.data, q), dtype=np.float64)
        mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T

        return self.controller.compute_torque(
            q=q,
            q_dot=q_dot,
            current_transform=current_transform,
            current_twist_body=current_twist,
            jacobian_body=jacobian,
            mass_matrix=mass_matrix,
            bias_forces=bias_forces,
            desired_transform=desired_transform,
            desired_twist_body=desired_twist_body,
            desired_twist_dot_body=desired_twist_dot_body,
            external_wrench_body=external_wrench_body,
        )


class PinocchioNominalSimulator:
    """Nominal forward dynamics used by the NRIC compensation loop."""

    def __init__(
        self,
        *,
        mjcf_path: str,
        end_effector_frame_name: str,
        time_step: float,
        gravity_z: float,
        include_joint_damping: bool = True,
    ) -> None:
        self.model = _build_nominal_pinocchio_model(
            mjcf_path=mjcf_path,
            gravity_z=gravity_z,
        )
        self.data = self.model.createData()
        self.frame_id = _lookup_frame_id(self.model, end_effector_frame_name)
        self.parent_joint_id = int(self.model.frames[self.frame_id].parentJoint)
        if self.parent_joint_id == 0:
            raise RuntimeError("Control frame must be attached to a non-root joint.")
        self.dt = float(time_step)
        self.include_joint_damping = bool(include_joint_damping)
        self.fext = [pin.Force.Zero() for _ in range(self.model.njoints)]
        self.q = np.zeros(NV_ARM, dtype=np.float64)
        self.dq = np.zeros(NV_ARM, dtype=np.float64)
        self.ddq = np.zeros(NV_ARM, dtype=np.float64)
        self.initialized = False

    def reset(self, q0: np.ndarray, dq0: Optional[np.ndarray] = None) -> None:
        self.q = _as_vector7(q0, "q0").copy()
        self.dq = np.zeros(NV_ARM, dtype=np.float64) if dq0 is None else _as_vector7(dq0, "dq0").copy()
        self.ddq = np.zeros(NV_ARM, dtype=np.float64)
        self._clear_external_forces()
        self.initialized = True

    def step(self, tau: np.ndarray, external_wrench_body: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("Call reset(q0) before stepping the nominal simulator.")

        tau = _as_vector7(tau, "tau").copy()
        external_wrench_body = (
            np.zeros(6, dtype=np.float64)
            if external_wrench_body is None
            else np.asarray(external_wrench_body, dtype=np.float64).reshape((6,))
        )

        self._clear_external_forces()
        pin.framesForwardKinematics(self.model, self.data, self.q)
        self.fext[self.parent_joint_id] = self._map_external_wrench_to_joint(external_wrench_body)

        tau_effective = tau.copy()
        if self.include_joint_damping:
            tau_effective -= np.asarray(self.model.damping, dtype=np.float64) * self.dq

        self.ddq = np.asarray(
            pin.aba(self.model, self.data, self.q, self.dq, tau_effective, self.fext),
            dtype=np.float64,
        )
        self.dq = self.dq + self.dt * self.ddq
        self.q = np.asarray(pin.integrate(self.model, self.q, self.dt * self.dq), dtype=np.float64)
        return self.q.copy(), self.dq.copy()

    def _clear_external_forces(self) -> None:
        for index in range(self.model.njoints):
            self.fext[index] = pin.Force.Zero()

    def _map_external_wrench_to_joint(self, external_wrench_body: np.ndarray) -> pin.Force:
        force_local = pin.Force(external_wrench_body[:3], external_wrench_body[3:])
        force_world = self.data.oMf[self.frame_id].act(force_local)
        return self.data.oMi[self.parent_joint_id].actInv(force_world)


@dataclass(frozen=True)
class NRICTorqueResult:
    joint_torque: np.ndarray
    nominal_torque_mujoco: np.ndarray
    nominal_torque_pinocchio: np.ndarray
    auxiliary_torque: np.ndarray
    auxiliary_torque_pd: np.ndarray
    auxiliary_torque_integral: np.ndarray
    nominal_q: np.ndarray
    nominal_qdot: np.ndarray


class NRICTorqueController:
    """Nominal Reference Internal Compensation (NRIC) wrapper."""

    def __init__(
        self,
        *,
        nominal_mjcf_path: str,
        end_effector_frame_name: str = "attachment",
        time_step: float = 1e-3,
        gravity_z: float = -9.81,
        enabled: bool = True,
        gic: Optional[SE3GeometryImpedance] = None,
        k_gamma: Optional[np.ndarray] = None,
        k_p: Optional[np.ndarray] = None,
        k_i: Optional[np.ndarray] = None,
        integral_limit: float = 0.2,
        integral_deadband: float = 5e-3,
        integral_leak_rate: float = 0.2,
        zero_cross_reset_ratio: float = 0.2,
        q_nominal: Optional[np.ndarray] = None,
        apparent_inertia: Optional[np.ndarray] = None,
        lambda_stiffness: Optional[np.ndarray] = None,
        physical_damping: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.dt = float(time_step)
        self.K_gamma = _as_diag7(k_gamma, default=2.0)
        self.K_p = _as_diag7(k_p, default=5.0)
        self.K_i = _as_diag7(k_i, default=0.1)
        self.integral_limit = float(integral_limit)
        self.integral_deadband = float(integral_deadband)
        self.integral_leak_rate = float(integral_leak_rate)
        self.zero_cross_reset_ratio = float(zero_cross_reset_ratio)
        self.integral_error = np.zeros(NV_ARM, dtype=np.float64)
        self.prev_error = np.zeros(NV_ARM, dtype=np.float64)
        self.initialized = False

        self.nominal_controller: Optional[PinocchioNominalSE3Controller] = None
        self.nominal_simulator: Optional[PinocchioNominalSimulator] = None
        if self.enabled:
            gic = gic or SE3GeometryImpedance()
            self.nominal_controller = PinocchioNominalSE3Controller(
                mjcf_path=nominal_mjcf_path,
                end_effector_frame_name=end_effector_frame_name,
                gravity_z=gravity_z,
                gic=gic,
                q_nominal=q_nominal,
                apparent_inertia=apparent_inertia,
                lambda_stiffness=lambda_stiffness,
                physical_damping=physical_damping,
                nullspace_stiffness=nullspace_stiffness,
                nullspace_damping=nullspace_damping,
            )
            self.nominal_simulator = PinocchioNominalSimulator(
                mjcf_path=nominal_mjcf_path,
                end_effector_frame_name=end_effector_frame_name,
                time_step=time_step,
                gravity_z=gravity_z,
                include_joint_damping=True,
            )

    def reset(self, q0: np.ndarray, dq0: Optional[np.ndarray] = None) -> None:
        q0 = _as_vector7(q0, "q0")
        dq0 = np.zeros(NV_ARM, dtype=np.float64) if dq0 is None else _as_vector7(dq0, "dq0")
        self.integral_error[:] = 0.0
        self.prev_error[:] = 0.0
        if self.enabled:
            assert self.nominal_simulator is not None
            self.nominal_simulator.reset(q0, dq0)
        self.initialized = True

    def get_internal_state(self) -> dict[str, np.ndarray | bool]:
        state = {
            "enabled": self.enabled,
            "initialized": self.initialized,
            "integral_error": self.integral_error.copy(),
            "prev_error": self.prev_error.copy(),
        }
        if self.enabled and self.nominal_simulator is not None:
            state["nominal_q"] = self.nominal_simulator.q.copy()
            state["nominal_qdot"] = self.nominal_simulator.dq.copy()
        return state

    def set_internal_state(self, state: dict[str, np.ndarray | bool]) -> None:
        self.initialized = bool(state.get("initialized", False))
        self.integral_error = np.asarray(state["integral_error"], dtype=np.float64).copy()
        self.prev_error = np.asarray(state["prev_error"], dtype=np.float64).copy()
        if self.enabled and self.nominal_simulator is not None and "nominal_q" in state:
            self.nominal_simulator.reset(
                np.asarray(state["nominal_q"], dtype=np.float64),
                np.asarray(state["nominal_qdot"], dtype=np.float64),
            )

    def compute_torque(
        self,
        *,
        q: np.ndarray,
        q_dot: np.ndarray,
        nominal_torque_mujoco: np.ndarray,
        desired_transform: np.ndarray,
        desired_twist_body: Optional[np.ndarray] = None,
        desired_twist_dot_body: Optional[np.ndarray] = None,
        external_wrench_body: Optional[np.ndarray] = None,
    ) -> NRICTorqueResult:
        q = _as_vector7(q, "q")
        q_dot = _as_vector7(q_dot, "q_dot")
        nominal_torque_mujoco = _as_vector7(nominal_torque_mujoco, "nominal_torque_mujoco")
        desired_transform = np.asarray(desired_transform, dtype=np.float64).reshape((4, 4))
        desired_twist_body = (
            np.zeros(6, dtype=np.float64)
            if desired_twist_body is None
            else np.asarray(desired_twist_body, dtype=np.float64).reshape((6,))
        )
        desired_twist_dot_body = (
            np.zeros(6, dtype=np.float64)
            if desired_twist_dot_body is None
            else np.asarray(desired_twist_dot_body, dtype=np.float64).reshape((6,))
        )
        external_wrench_body = (
            np.zeros(6, dtype=np.float64)
            if external_wrench_body is None
            else np.asarray(external_wrench_body, dtype=np.float64).reshape((6,))
        )

        if not self.enabled:
            zeros = np.zeros(NV_ARM, dtype=np.float64)
            return NRICTorqueResult(
                joint_torque=nominal_torque_mujoco,
                nominal_torque_mujoco=nominal_torque_mujoco.copy(),
                nominal_torque_pinocchio=zeros.copy(),
                auxiliary_torque=zeros.copy(),
                auxiliary_torque_pd=zeros.copy(),
                auxiliary_torque_integral=zeros.copy(),
                nominal_q=q.copy(),
                nominal_qdot=q_dot.copy(),
            )

        if not self.initialized:
            self.reset(q, q_dot)

        assert self.nominal_controller is not None
        assert self.nominal_simulator is not None

        nominal_q = self.nominal_simulator.q.copy()
        nominal_qdot = self.nominal_simulator.dq.copy()
        nominal_result = self.nominal_controller.compute_torque(
            q=nominal_q,
            q_dot=nominal_qdot,
            desired_transform=desired_transform,
            desired_twist_body=desired_twist_body,
            desired_twist_dot_body=desired_twist_dot_body,
            external_wrench_body=external_wrench_body,
        )

        e_nr = nominal_q - q
        e_nr_dot = nominal_qdot - q_dot
        self._update_integral_error(e_nr)

        tau_aux_integral = self.K_gamma @ (self.K_i @ self.integral_error)
        tau_aux_pd = self.K_gamma @ (e_nr_dot + self.K_p @ e_nr)
        tau_aux = tau_aux_pd + tau_aux_integral

        self.nominal_simulator.step(
            nominal_result.joint_torque,
            external_wrench_body=external_wrench_body,
        )

        return NRICTorqueResult(
            joint_torque=nominal_torque_mujoco + tau_aux,
            nominal_torque_mujoco=nominal_torque_mujoco.copy(),
            nominal_torque_pinocchio=nominal_result.joint_torque.copy(),
            auxiliary_torque=tau_aux.copy(),
            auxiliary_torque_pd=tau_aux_pd.copy(),
            auxiliary_torque_integral=tau_aux_integral.copy(),
            nominal_q=nominal_q,
            nominal_qdot=nominal_qdot,
        )

    def _update_integral_error(self, e_nr: np.ndarray) -> None:
        leak = np.clip(1.0 - self.integral_leak_rate * self.dt, 0.0, 1.0)
        self.integral_error *= leak
        zero_cross_mask = self.prev_error * e_nr < 0.0
        if np.any(zero_cross_mask):
            self.integral_error[zero_cross_mask] *= self.zero_cross_reset_ratio

        active_mask = np.abs(e_nr) >= self.integral_deadband
        self.integral_error[active_mask] += self.dt * e_nr[active_mask]
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit,
        )
        self.prev_error = e_nr.copy()
