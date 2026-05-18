"""
Configuration helpers for the MuJoCo Franka insertion task.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from serl_robot_infra.envs.franka_insertion_env import InsertionConfig

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "franka_panda"

DEFAULT_RESET_POS_NOISE_LOW = np.array([-0.012, -0.003, 0.0], dtype=np.float64)
DEFAULT_RESET_POS_NOISE_HIGH = np.array([0.012, 0.003, 0.0], dtype=np.float64)
DEFAULT_RESET_YAW_RANGE = 0.12
DEFAULT_EVAL_CASE_XY_LOW = np.array([-0.012, -0.003], dtype=np.float64)
DEFAULT_EVAL_CASE_XY_HIGH = np.array([0.012, 0.003], dtype=np.float64)
DEFAULT_EVAL_CASE_YAW_RANGE = 0.12
WIDE_EVAL_CASE_XY_LOW = np.array([-0.01, -0.01], dtype=np.float64)
WIDE_EVAL_CASE_XY_HIGH = np.array([0.01, 0.01], dtype=np.float64)
WIDE_EVAL_CASE_YAW_RANGE = 0.25


def build_reset_perturbation_overrides(
    *,
    preset: str = "default",
    yaw_range_override: Optional[float] = None,
    zero_reset_perturbation: bool = False,
) -> dict:
    """Return config overrides for reset perturbation presets."""
    if zero_reset_perturbation:
        return {
            "reset_pos_noise_low": np.zeros(3, dtype=np.float64),
            "reset_pos_noise_high": np.zeros(3, dtype=np.float64),
            "reset_yaw_range": 0.0,
        }

    overrides: dict = {}
    if preset == "yaw_only":
        overrides.update(
            reset_pos_noise_low=np.zeros(3, dtype=np.float64),
            reset_pos_noise_high=np.zeros(3, dtype=np.float64),
        )
    elif preset != "default":
        raise ValueError(f"Unknown reset perturbation preset: {preset}")

    if yaw_range_override is not None:
        overrides["reset_yaw_range"] = float(yaw_range_override)

    return overrides


def build_eval_task_case_overrides(
    *,
    preset: str = "aligned",
    yaw_range_override: Optional[float] = None,
) -> dict:
    """Return config overrides for eval task-case presets."""
    overrides: dict = {}
    if preset == "aligned":
        pass
    elif preset == "wide":
        overrides.update(
            eval_case_xy_low=WIDE_EVAL_CASE_XY_LOW.copy(),
            eval_case_xy_high=WIDE_EVAL_CASE_XY_HIGH.copy(),
            eval_case_yaw_range=WIDE_EVAL_CASE_YAW_RANGE,
        )
    elif preset == "yaw_only":
        overrides.update(
            eval_case_xy_low=np.zeros(2, dtype=np.float64),
            eval_case_xy_high=np.zeros(2, dtype=np.float64),
        )
    else:
        raise ValueError(f"Unknown eval task-case preset: {preset}")

    if yaw_range_override is not None:
        overrides["eval_case_yaw_range"] = float(yaw_range_override)

    return overrides


def make_insertion_config(**overrides) -> InsertionConfig:
    """Create the default insertion-task configuration."""
    defaults = dict(
        mjcf_path=str(_MODEL_DIR / "insertion_scene_effort.xml"),
        nominal_pinocchio_mjcf_path=str(_MODEL_DIR / "panda_nohand.xml"),
        ee_site_name="tool_tip_site",
        socket_site_name="tool_socket_center_site",
        task_frame_site_name="tool_socket_center_site",
        control_frame_site_name="tool_mount",
        nominal_pinocchio_frame_name="attachment",
        home_key_name="home",
        actuator_mode="torque",

        sim_timestep=0.001,
        n_substeps=10,
        control_dt=0.01,
        enable_gravity_compensation=False,

        pos_action_scale=0.02,
        rot_action_scale=0.08,

        reset_pos_noise_low=DEFAULT_RESET_POS_NOISE_LOW.copy(),
        reset_pos_noise_high=DEFAULT_RESET_POS_NOISE_HIGH.copy(),
        reset_yaw_range=DEFAULT_RESET_YAW_RANGE,
        eval_case_xy_low=DEFAULT_EVAL_CASE_XY_LOW.copy(),
        eval_case_xy_high=DEFAULT_EVAL_CASE_XY_HIGH.copy(),
        eval_case_yaw_range=DEFAULT_EVAL_CASE_YAW_RANGE,
        reset_ik_max_iters=80,
        reset_ik_tolerance=1e-4,
        reset_ik_position_tolerance=1e-4,
        reset_ik_rotation_tolerance=1e-3,
        reset_ik_position_weight=1.0,
        reset_ik_rotation_weight=0.05,
        reset_ik_seed_weight=0.01,
        reset_ik_center_weight=0.002,

        reward_body_pos_weight=1.0,
        reward_body_rot_weight=3.0,
        reward_linear_speed_weight=0.15,
        reward_angular_speed_weight=0.10,
        reward_lateral_force_weight=0.20,
        reward_lateral_torque_weight=0.10,
        reward_depth_weight=2.0,
        reward_success_bonus=2000.0,
        reward_failure_penalty=25.0,
        reward_step_penalty=0.01,
        reward_depth_xy_threshold=0.004,
        reward_depth_rot_threshold=0.12,
        reward_depth_deadband=0.0002,

        success_xy_threshold=0.002,
        success_rot_threshold=0.1,
        success_z_gap_threshold=0.002,
        success_linear_velocity_threshold=0.01,
        success_angular_velocity_threshold=0.05,

        workspace_low=np.array([0.49, -0.05, -0.02]),
        workspace_high=np.array([0.62, 0.05, 0.12]),
        failure_xy_threshold=0.012,
        failure_z_gap_threshold=0.012,

        input_deadzone=0.05,
        input_filter_alpha=0.25,
        max_reference_linear_velocity=0.05,
        max_reference_angular_velocity=0.10,
        max_reference_linear_offset=0.01,
        max_reference_angular_offset=0.2,
        wrench_force_scale=20.0,
        wrench_torque_scale=1.0,
        ik_damping=1e-4,
        joint_velocity_limits=(2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610),
        joint_limit_margin=0.05,
        use_external_wrench_feedback=False,
        enable_nric=True,
        impedance_inertia=(1.0, 1.0, 1.0, 0.1, 0.1, 0.1),
        impedance_stiffness=(100.0, 100.0, 100.0, 10.0, 10.0, 10.0),
        impedance_damping=(
            10.0,
            10.0,
            10.0,
            1.0,
            1.0,
            1.0,
        ),
        nullspace_stiffness=10.0,
        nullspace_damping=2.0,
        nric_k_gamma=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        nric_k_p=(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
        nric_k_i=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        nric_integral_limit=0.2,
        nric_integral_deadband=5e-3,
        nric_integral_leak_rate=0.2,
        nric_zero_cross_reset_ratio=0.2,

        max_episode_steps=3000,
        render_camera="front",
    )
    defaults.update(overrides)
    return InsertionConfig(**defaults)
