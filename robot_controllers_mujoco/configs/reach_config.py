"""
Configuration for the Franka reach task training.
"""

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from robot_controllers_mujoco.envs.franka_reach_env import ReachConfig

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "franka_panda"


def make_reach_config(**overrides) -> ReachConfig:
    """Create a default ReachConfig with the correct MJCF path."""
    defaults = dict(
        mjcf_path=str(_MODEL_DIR / "reach_scene.xml"),

        # Simulation timing
        sim_timestep=0.002,
        n_substeps=20,

        # Action scaling
        pos_action_scale=0.05,
        rot_action_scale=0.1,

        # Target volume
        target_pos_low=np.array([0.25, -0.35, 0.15]),
        target_pos_high=np.array([0.65, 0.35, 0.65]),
        target_rot_range=0.5,

        # Reward
        reward_pos_weight=1.0,
        reward_rot_weight=0.3,
        reward_success_bonus=10.0,

        # Thresholds
        success_pos_threshold=0.02,
        success_rot_threshold=0.15,

        # Episode
        max_episode_steps=200,

        # Render
        render_camera="front",
    )
    defaults.update(overrides)
    return ReachConfig(**defaults)


@dataclass
class TrainConfig:
    """Hyperparameters for SAC training."""

    seed: int = 42
    total_timesteps: int = 500_000
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    initial_temperature: float = 0.1
    learning_starts: int = 5000
    train_freq: int = 1
    gradient_steps: int = 1

    hidden_dims: tuple = (256, 256)

    log_interval: int = 1000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    save_interval: int = 50_000

    output_dir: str = "output/reach_training"
