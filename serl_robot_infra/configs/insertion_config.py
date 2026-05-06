"""
Configuration helpers for the MuJoCo Franka insertion task.
"""

from pathlib import Path

import numpy as np

from serl_robot_infra.envs.franka_insertion_env import InsertionConfig

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "franka_panda"


def make_insertion_config(**overrides) -> InsertionConfig:
    """Create the default insertion-task configuration."""
    defaults = dict(
        mjcf_path=str(_MODEL_DIR / "insertion_scene.xml"),
        ee_site_name="tool_tip_site",
        socket_site_name="tool_socket_center_site",
        home_key_name="home",
        actuator_mode="velocity",

        sim_timestep=0.001,
        n_substeps=10,
        control_dt=0.01,
        enable_gravity_compensation=True,

        pos_action_scale=0.02,
        rot_action_scale=0.08,

        reset_pos_noise_low=np.array([-0.01, -0.0015, 0.0]),
        reset_pos_noise_high=np.array([0.01, 0.0015, 0.0]),
        reset_yaw_range=0.08,
        reset_ik_max_iters=80,
        reset_ik_tolerance=1e-4,
        reset_ik_position_tolerance=1e-4,
        reset_ik_rotation_tolerance=1e-3,
        reset_ik_position_weight=1.0,
        reset_ik_rotation_weight=0.05,
        reset_ik_seed_weight=0.01,
        reset_ik_center_weight=0.002,

        reward_xy_weight=8.0,
        reward_z_weight=1.5,
        reward_rot_weight=0.3,
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
        pose_error_gain=25.0,
        ik_damping=1e-4,
        nullspace_gain=0.15,
        joint_velocity_limits=(2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610),
        joint_limit_margin=0.05,
        enable_admittance=False,
        wrench_filter_alpha=0.2,
        force_deadband=(1.0, 1.0, 1.0),
        torque_deadband=(0.05, 0.05, 0.05),
        max_force_world=(20.0, 20.0, 20.0),
        max_torque_world=(2.0, 2.0, 2.0),
        admittance_mass=(2.0, 2.0, 3.0, 0.15, 0.15, 0.15),
        admittance_damping=(80.0, 80.0, 110.0, 8.0, 8.0, 8.0),
        admittance_stiffness=(80.0, 80.0, 140.0, 10.0, 10.0, 10.0),
        max_compliance_linear_offset=0.02,
        max_compliance_angular_offset=0.2,
        max_compliance_linear_velocity=0.10,
        max_compliance_angular_velocity=0.5,
        joint_velocity_filter_alpha=1.0,

        max_episode_steps=1000,
        render_camera="front",
    )
    defaults.update(overrides)
    return InsertionConfig(**defaults)
