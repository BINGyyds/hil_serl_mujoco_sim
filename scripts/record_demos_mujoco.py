#!/usr/bin/env python3
"""
Record demonstrations in MuJoCo using SpaceMouse.

The operator controls the Franka arm end-effector with the SpaceMouse while
a MuJoCo viewer shows the simulation.  Demonstrations are saved as a list of
transitions compatible with the RLPD demo buffer.

Button mapping:
    Right button  -- save current episode as a successful demo
    Left  button  -- discard current episode (start over)

Usage:
    python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl
    python scripts/record_demos_mujoco.py --output demo_data/reach_demos.pkl --num_demos 10
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_controllers_mujoco.configs.reach_config import make_reach_config
from robot_controllers_mujoco.envs.franka_reach_env import FrankaReachEnv
from robot_controllers_mujoco.envs.wrappers import MujocoSERLObsWrapper
from robot_controllers_mujoco.spacemouse.spacemouse_expert import SpaceMouseExpert


def main():
    parser = argparse.ArgumentParser(description="Record MuJoCo demos with SpaceMouse")
    parser.add_argument(
        "--output", type=str, required=True, help="Output pickle path"
    )
    parser.add_argument(
        "--num_demos", type=int, default=10, help="Number of demos to record"
    )
    parser.add_argument(
        "--action_scale", type=float, default=1.0,
        help="Scale factor for SpaceMouse actions",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    proprio_keys = ["tcp_pose", "tcp_vel", "q_pos", "q_vel", "target_pose"]

    cfg = make_reach_config()
    env = FrankaReachEnv(config=cfg, render_mode="human")
    env = MujocoSERLObsWrapper(env, proprio_keys=proprio_keys)

    expert = SpaceMouseExpert()

    all_demos: list = []
    episode_transitions: list = []
    demos_saved = 0

    print(f"Recording {args.num_demos} demonstrations.")
    print("  SpaceMouse controls the end-effector (6 DOF).")
    print("  RIGHT button = save episode | LEFT button = discard episode")
    print("-" * 60)

    obs, _ = env.reset()
    env.render()

    try:
        while demos_saved < args.num_demos:
            expert_a, buttons = expert.get_action()
            left_pressed = bool(buttons[0])
            right_pressed = bool(buttons[1])

            action = expert_a[:6] * args.action_scale
            next_obs, reward, done, truncated, info = env.step(action)
            env.render()

            transition = dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            episode_transitions.append(transition)
            obs = next_obs

            if right_pressed or (done and info.get("success", False)):
                all_demos.extend(episode_transitions)
                demos_saved += 1
                print(
                    f"  [SAVED] Demo {demos_saved}/{args.num_demos} "
                    f"({len(episode_transitions)} steps)"
                )
                episode_transitions = []
                obs, _ = env.reset()
                env.render()
                time.sleep(0.3)

            elif left_pressed or done or truncated:
                print(
                    f"  [DISCARDED] Episode ({len(episode_transitions)} steps)"
                )
                episode_transitions = []
                obs, _ = env.reset()
                env.render()
                time.sleep(0.3)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        expert.close()
        env.close()

    if all_demos:
        with open(args.output, "wb") as f:
            pkl.dump(all_demos, f)
        print(f"\nSaved {demos_saved} demos ({len(all_demos)} transitions) to {args.output}")
    else:
        print("\nNo demos saved.")


if __name__ == "__main__":
    main()
