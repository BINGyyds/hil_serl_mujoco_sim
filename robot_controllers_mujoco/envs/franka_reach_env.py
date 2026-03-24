"""
Franka Panda reach task: move end-effector to a random 6D target pose.

The target is visualised via a mocap body in the scene, and the reward is
a weighted combination of position error, orientation error, and a sparse
success bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from robot_controllers_mujoco.envs.franka_mujoco_env import (
    FrankaMujocoConfig,
    FrankaMujocoEnv,
)


@dataclass
class ReachConfig(FrankaMujocoConfig):
    """Extra configuration for the reach task."""

    # Target sampling volume (world frame, metres)
    target_pos_low: np.ndarray = field(
        default_factory=lambda: np.array([0.25, -0.35, 0.15])
    )
    target_pos_high: np.ndarray = field(
        default_factory=lambda: np.array([0.65, 0.35, 0.65])
    )

    # Target orientation sampling: random euler perturbation around home EE orientation
    target_rot_range: float = 0.5   # max radians per euler axis

    # Reward weights
    reward_pos_weight: float = 1.0
    reward_rot_weight: float = 0.3
    reward_success_bonus: float = 10.0

    # Success thresholds
    success_pos_threshold: float = 0.02    # metres
    success_rot_threshold: float = 0.15    # radians

    # Episode length
    max_episode_steps: int = 200

    # Mocap body name in MJCF for target visualisation
    target_mocap_name: str = "target"

    # Whether to use fixed target (for debugging) instead of random
    fixed_target_pos: Optional[np.ndarray] = None
    fixed_target_euler: Optional[np.ndarray] = None


class FrankaReachEnv(FrankaMujocoEnv):
    """
    Task: move the Franka end-effector to a randomised 6D target pose.

    The target is shown as a red sphere (mocap body) in the scene.
    The episode terminates on success or after ``max_episode_steps``.
    """

    def __init__(
        self,
        config: Optional[ReachConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.reach_cfg = config or ReachConfig()
        super().__init__(config=self.reach_cfg, render_mode=render_mode)

        # Mocap body for target visualisation
        self._target_mocap_id: Optional[int] = None
        try:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.reach_cfg.target_mocap_name
            )
            if body_id >= 0:
                self._target_mocap_id = body_id
        except Exception:
            pass

        # Target pose (set during reset)
        self._target_pos = np.zeros(3)
        self._target_rot = np.eye(3)  # rotation matrix

        self._step_count = 0

        # Extend observation space with target_pose
        state_spaces = dict(self._base_state_spaces)
        state_spaces["target_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float64
        )
        self.observation_space = spaces.Dict({
            "state": spaces.Dict(state_spaces),
        })

        # Cache the home EE orientation for sampling perturbations
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addrs] = self.cfg.home_qpos.copy()
        self.data.ctrl[self._actuator_ids] = self.cfg.home_qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        _, self._home_ee_rot = self._get_ee_pose()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        obs, info = super().reset(seed=seed, options=options)
        self._step_count = 0
        self._sample_target()
        self._update_target_visual()

        obs = self._get_obs()
        info = self._get_info(obs)
        return obs, info

    def _sample_target(self) -> None:
        """Sample a random reachable target pose."""
        cfg = self.reach_cfg

        if cfg.fixed_target_pos is not None:
            self._target_pos = cfg.fixed_target_pos.copy()
        else:
            self._target_pos = self.np_random.uniform(
                cfg.target_pos_low, cfg.target_pos_high
            )

        if cfg.fixed_target_euler is not None:
            self._target_rot = Rotation.from_euler("xyz", cfg.fixed_target_euler).as_matrix()
        else:
            euler_delta = self.np_random.uniform(
                -cfg.target_rot_range, cfg.target_rot_range, size=3
            )
            delta_rot = Rotation.from_euler("xyz", euler_delta).as_matrix()
            self._target_rot = delta_rot @ self._home_ee_rot

    def _update_target_visual(self) -> None:
        """Move the mocap body to show the current target."""
        if self._target_mocap_id is None:
            return

        # Find the mocap index for this body
        mocap_id = self.model.body_mocapid[self._target_mocap_id]
        if mocap_id < 0:
            return

        self.data.mocap_pos[mocap_id] = self._target_pos
        self.data.mocap_quat[mocap_id] = Rotation.from_matrix(self._target_rot).as_quat(scalar_first=True)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1

        if self._step_count >= self.reach_cfg.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation (augment with target info)
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        obs = super()._get_obs()
        target_quat = Rotation.from_matrix(self._target_rot).as_quat()  # xyzw
        obs["state"]["target_pose"] = np.concatenate([self._target_pos, target_quat])
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: Dict) -> float:
        ee_pos, ee_rot = self._get_ee_pose()
        cfg = self.reach_cfg

        pos_error = np.linalg.norm(ee_pos - self._target_pos)

        # Rotation error: angle of R_err = R_target^T @ R_current
        R_err = self._target_rot.T @ ee_rot
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))

        reward = (
            -cfg.reward_pos_weight * pos_error
            - cfg.reward_rot_weight * angle_err
        )

        if pos_error < cfg.success_pos_threshold and angle_err < cfg.success_rot_threshold:
            reward += cfg.reward_success_bonus

        return float(reward)

    def _check_termination(self, obs: Dict) -> bool:
        ee_pos, ee_rot = self._get_ee_pose()
        cfg = self.reach_cfg
        pos_error = np.linalg.norm(ee_pos - self._target_pos)
        R_err = self._target_rot.T @ ee_rot
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        return bool(pos_error < cfg.success_pos_threshold and angle_err < cfg.success_rot_threshold)

    def _get_info(self, obs: Dict) -> Dict[str, Any]:
        ee_pos, ee_rot = self._get_ee_pose()
        pos_error = np.linalg.norm(ee_pos - self._target_pos)
        R_err = self._target_rot.T @ ee_rot
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        success = pos_error < self.reach_cfg.success_pos_threshold and angle_err < self.reach_cfg.success_rot_threshold
        return {
            "pos_error": pos_error,
            "rot_error": angle_err,
            "success": success,
        }
