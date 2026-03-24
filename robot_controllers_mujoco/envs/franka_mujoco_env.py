"""
Base MuJoCo Gymnasium environment for Franka Panda.

Provides Cartesian delta control mapped to joint-level position actuators
via the site Jacobian pseudo-inverse.  Observation and action interfaces
are designed to be compatible with the hil-serl training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

@dataclass
class FrankaMujocoConfig:
    """All tunables for :class:`FrankaMujocoEnv`."""

    mjcf_path: str = ""

    # Joint names in the MJCF
    arm_joint_names: Tuple[str, ...] = tuple(f"joint{i+1}" for i in range(7))
    arm_actuator_names: Tuple[str, ...] = tuple(f"actuator{i+1}" for i in range(7))
    ee_site_name: str = "attachment_site"

    # Home joint configuration (Panda default)
    home_qpos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854])
    )

    # Action scaling
    pos_action_scale: float = 0.05      # metres per unit action
    rot_action_scale: float = 0.1       # radians per unit action

    # Simulation
    n_substeps: int = 20                 # mj_step calls per env.step()
    sim_timestep: float = 0.002          # MuJoCo dt
    control_dt: float = 0.04            # = n_substeps * sim_timestep  (25 Hz)

    # Rendering
    render_width: int = 640
    render_height: int = 480
    render_camera: str = "front"

    # Jacobian damping factor for pseudo-inverse (avoids singularity)
    jac_damping: float = 1e-4


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FrankaMujocoEnv(gym.Env):
    """
    MuJoCo-based Gymnasium environment for the Franka Panda robot.

    **Action space** (6-D, continuous, [-1, 1]):
        [dx, dy, dz, drx, dry, drz]
        Cartesian end-effector delta in the world frame.

    **Observation space** (dict):
        state/tcp_pose   (7,)  xyz + quaternion (xyzw)
        state/tcp_vel    (6,)  linear + angular velocity
        state/q_pos      (7,)  joint angles
        state/q_vel      (7,)  joint velocities
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        config: Optional[FrankaMujocoConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or FrankaMujocoConfig()
        self.render_mode = render_mode

        # --- Load MuJoCo model ---
        self.model = mujoco.MjModel.from_xml_path(self.cfg.mjcf_path)
        self.model.opt.timestep = self.cfg.sim_timestep
        self.data = mujoco.MjData(self.model)

        # --- Cache IDs ---
        self._joint_ids = np.array(
            [self.model.joint(n).id for n in self.cfg.arm_joint_names], dtype=np.int32
        )
        self._qpos_addrs = self.model.jnt_qposadr[self._joint_ids]
        self._dof_addrs = self.model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = np.array(
            [self.model.actuator(n).id for n in self.cfg.arm_actuator_names], dtype=np.int32
        )
        self._ee_site_id = self.model.site(self.cfg.ee_site_name).id

        self._nq = len(self._joint_ids)  # 7

        # Joint limits from the model
        limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        self._q_low = np.where(limited, self.model.jnt_range[self._joint_ids, 0], -2 * np.pi)
        self._q_high = np.where(limited, self.model.jnt_range[self._joint_ids, 1], 2 * np.pi)

        # --- Action / observation spaces ---
        self.action_space = spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
        )

        self._base_state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "tcp_vel":  spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "q_pos":    spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "q_vel":    spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
        }
        self.observation_space = spaces.Dict({
            "state": spaces.Dict(self._base_state_spaces),
        })

        # --- Renderer (lazy init) ---
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer: Any = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self._qpos_addrs] = self.cfg.home_qpos.copy()
        self.data.ctrl[self._actuator_ids] = self.cfg.home_qpos.copy()
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        cart_delta = np.zeros(6, dtype=np.float64)
        cart_delta[:3] = action[:3] * self.cfg.pos_action_scale
        cart_delta[3:] = action[3:] * self.cfg.rot_action_scale

        q_current = self.data.qpos[self._qpos_addrs].copy()
        dq = self._cartesian_to_joint(cart_delta)
        q_target = np.clip(q_current + dq, self._q_low, self._q_high)

        self.data.ctrl[self._actuator_ids] = q_target

        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = False
        info = self._get_info(obs)

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
            return None
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # State access helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        q_pos = self.data.qpos[self._qpos_addrs].copy()
        q_vel = self.data.qvel[self._dof_addrs].copy()

        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        ee_rot_mat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        ee_quat = Rotation.from_matrix(ee_rot_mat).as_quat()  # xyzw

        # Compute EE velocity via site Jacobian
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        full_qvel = self.data.qvel.copy()
        ee_lin_vel = jacp @ full_qvel
        ee_ang_vel = jacr @ full_qvel
        tcp_vel = np.concatenate([ee_lin_vel, ee_ang_vel])

        tcp_pose = np.concatenate([ee_pos, ee_quat])

        return {
            "state": {
                "tcp_pose": tcp_pose,
                "tcp_vel": tcp_vel,
                "q_pos": q_pos,
                "q_vel": q_vel,
            }
        }

    def _get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position[3], rotation_matrix[3,3]) of end-effector."""
        pos = self.data.site_xpos[self._ee_site_id].copy()
        rot = self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
        return pos, rot

    # ------------------------------------------------------------------
    # Cartesian → Joint conversion
    # ------------------------------------------------------------------

    def _cartesian_to_joint(self, cart_delta: np.ndarray) -> np.ndarray:
        """
        Convert a 6D Cartesian displacement to joint displacement
        using the damped pseudo-inverse of the site Jacobian.
        """
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)

        # Extract columns for our arm DOFs only
        J = np.vstack([
            jacp[:, self._dof_addrs],
            jacr[:, self._dof_addrs],
        ])  # (6, 7)

        # Damped least-squares pseudo-inverse
        lam = self.cfg.jac_damping
        JJT = J @ J.T + lam * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, cart_delta)
        return dq

    # ------------------------------------------------------------------
    # Reward / termination (override in subclass)
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: Dict) -> float:
        return 0.0

    def _check_termination(self, obs: Dict) -> bool:
        return False

    def _get_info(self, obs: Dict) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, self.cfg.render_height, self.cfg.render_width
            )
        self._renderer.update_scene(self.data, camera=self.cfg.render_camera)
        return self._renderer.render()

    def _render_human(self) -> None:
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
