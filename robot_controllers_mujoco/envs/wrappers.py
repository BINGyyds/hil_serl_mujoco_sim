"""
Wrappers for MuJoCo simulation environments.

Provides SpaceMouse intervention and observation flattening that are compatible
with the hil-serl training pipeline but work with state-only MuJoCo envs
(no real-hardware HTTP calls, no images required).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten, flatten_space


class MujocoSpacemouseIntervention(gym.ActionWrapper):
    """SpaceMouse human intervention for 6-D MuJoCo environments (no gripper).

    During ``step()``, the wrapper reads the SpaceMouse.  If the expert
    provides a non-zero input the expert action replaces the policy action and
    ``info["intervene_action"]`` is set so the training loop can record it.

    Button behaviour:
        * **Left button** -- mark the current episode as failed and force
          ``terminated = True`` so the environment resets.
        * **Right button** -- (reserved, no-op for now).
    """

    def __init__(self, env: gym.Env, action_scale: float = 1.0):
        super().__init__(env)
        assert env.action_space.shape == (6,), (
            f"Expected 6-D action space, got {env.action_space.shape}"
        )
        from robot_controllers_mujoco.spacemouse.spacemouse_expert import (
            SpaceMouseExpert,
        )

        self.expert = SpaceMouseExpert()
        self.left: bool = False
        self.right: bool = False
        self.action_scale = action_scale
        self._cached_action: Optional[np.ndarray] = None
        self._cached_replaced: Optional[bool] = None

    def _read_expert_action(self) -> Tuple[np.ndarray, bool]:
        expert_a, buttons = self.expert.get_action()
        self.left = bool(buttons[0]) if len(buttons) > 0 else False
        self.right = bool(buttons[1]) if len(buttons) > 1 else False
        expert_a = expert_a[:6] * self.action_scale

        intervened = np.linalg.norm(expert_a) > 0.001 or self.left or self.right
        return expert_a, intervened

    def peek_intervention(self) -> Tuple[np.ndarray, bool]:
        """Read SpaceMouse once and cache the result for the next ``step`` call."""
        if self._cached_action is None or self._cached_replaced is None:
            self._cached_action, self._cached_replaced = self._read_expert_action()
        return self._cached_action.copy(), bool(self._cached_replaced)

    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self._cached_action is not None and self._cached_replaced is not None:
            expert_a = self._cached_action
            intervened = self._cached_replaced
            self._cached_action = None
            self._cached_replaced = None
        else:
            expert_a, intervened = self._read_expert_action()

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        if self.left:
            done = True
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._cached_action = None
        self._cached_replaced = None
        self.left = False
        self.right = False
        return super().reset(**kwargs)

    def close(self):
        self.expert.close()
        super().close()


class MujocoSERLObsWrapper(gym.ObservationWrapper):
    """Flatten nested ``state`` dict for state-only MuJoCo environments.

    This is the state-only counterpart of
    :class:`serl_launcher.wrappers.serl_obs_wrappers.SERLObsWrapper` which
    assumes an ``images`` key in the observation.  Here we only keep ``state``.
    """

    def __init__(self, env: gym.Env, proprio_keys: Optional[List[str]] = None):
        super().__init__(env)
        self.proprio_keys = proprio_keys or list(
            self.env.observation_space["state"].keys()
        )
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        self.observation_space = gym.spaces.Dict(
            {"state": flatten_space(self.proprio_space)}
        )

    def observation(self, obs):
        return {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
