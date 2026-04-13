#!/usr/bin/env python3
"""
Record MuJoCo peg-in-hole insertion demonstrations with the SpaceMouse.

Recommended workflow:
  1. Record 20-50 successful demonstrations.
  2. Use the resulting pickle as ``--demo_path`` for the learner.
  3. Continue with online HIL-RLPD on top of this demo-prefilled buffer.
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_controllers_mujoco.configs.insertion_config import make_insertion_config
from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv
from robot_controllers_mujoco.envs.wrappers import MujocoSERLObsWrapper
from robot_controllers_mujoco.spacemouse.spacemouse_expert import SpaceMouseExpert


def make_expert():
    """Create the SpaceMouse expert used for real demo recording."""
    return SpaceMouseExpert()


def maybe_render(env, render_enabled: bool) -> None:
    """Render one frame only when explicitly enabled."""
    if render_enabled:
        env.render()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record MuJoCo insertion demos with the SpaceMouse."
    )
    parser.add_argument("--output", type=str, required=True, help="Output pickle path.")
    parser.add_argument(
        "--num_demos",
        type=int,
        default=25,
        help="Number of successful demos to record.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Extra scale applied to SpaceMouse actions.",
    )
    parser.add_argument(
        "--render-human",
        action="store_true",
        help="显式打开 MuJoCo viewer。",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按 env.control_dt 近似做实时节奏推进，主要用于配合 --render-human 观察。",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="单条 demo 最多允许的环境步数。默认 1000 步约 40 秒。",
    )
    parser.add_argument(
        "--debug-spacemouse",
        action="store_true",
        help="周期性打印 SpaceMouse 动作和按钮，用于确认输入是否生效。",
    )
    parser.add_argument(
        "--debug-interval",
        type=float,
        default=0.5,
        help="--debug-spacemouse 的打印间隔秒数。",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "q_pos",
        "q_vel",
        "socket_pose",
        "relative_pose",
        "insertion_depth",
        "controller_target_pose",
    ]

    render_mode: Optional[str] = "human" if args.render_human else None
    raw_env = FrankaInsertionEnv(
        config=make_insertion_config(max_episode_steps=args.max_episode_steps),
        render_mode=render_mode,
    )
    env = MujocoSERLObsWrapper(raw_env, proprio_keys=proprio_keys)
    expert = make_expert()

    all_demos: list = []
    episode_transitions: list = []
    demos_saved = 0

    print(f"Recording {args.num_demos} insertion demonstrations.")
    print("  SpaceMouse controls the tool tip (6 DOF).")
    print("  RIGHT button = save successful episode | LEFT button = discard/reset")
    print(f"  Max episode steps: {args.max_episode_steps}")
    print(f"  Action scale: {args.action_scale}")
    if args.render_human:
        print("  Human viewer enabled.")
    else:
        print("  Running headless.")
    print("-" * 60)

    obs, _ = env.reset()
    maybe_render(env, args.render_human)
    last_debug_print = 0.0

    try:
        while demos_saved < args.num_demos:
            step_start = time.perf_counter()
            expert_action, buttons = expert.get_action()
            left_pressed = bool(buttons[0]) if len(buttons) > 0 else False
            right_pressed = bool(buttons[1]) if len(buttons) > 1 else False

            action = np.clip(expert_action[:6] * args.action_scale, -1.0, 1.0)
            if (
                args.debug_spacemouse
                and step_start - last_debug_print >= args.debug_interval
            ):
                action_text = np.array2string(
                    action,
                    precision=3,
                    suppress_small=True,
                )
                print(f"  [INPUT] action={action_text} buttons={buttons}", flush=True)
                last_debug_print = step_start

            next_obs, reward, done, truncated, info = env.step(action)
            maybe_render(env, args.render_human)

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

            should_save = right_pressed or (done and info.get("success", False))
            should_discard = left_pressed or done or truncated

            if should_save:
                if episode_transitions:
                    episode_transitions[-1]["dones"] = True
                    episode_transitions[-1]["masks"] = 0.0
                all_demos.extend(episode_transitions)
                demos_saved += 1
                print(
                    f"  [SAVED] Demo {demos_saved}/{args.num_demos} "
                    f"({len(episode_transitions)} steps, "
                    f"xy_error={info.get('xy_error', 0.0):.4f}, "
                    f"z_gap={info.get('z_gap', 0.0):.4f}, "
                    f"rot_error={info.get('rot_error', 0.0):.4f})"
                )
                episode_transitions = []
                obs, _ = env.reset()
                maybe_render(env, args.render_human)
                time.sleep(0.3)
            elif should_discard:
                print(
                    f"  [DISCARDED] Episode ({len(episode_transitions)} steps, "
                    f"success={info.get('success', False)}, "
                    f"xy_error={info.get('xy_error', 0.0):.4f}, "
                    f"z_gap={info.get('z_gap', 0.0):.4f}, "
                    f"rot_error={info.get('rot_error', 0.0):.4f})"
                )
                episode_transitions = []
                obs, _ = env.reset()
                maybe_render(env, args.render_human)
                time.sleep(0.3)

            if args.realtime:
                elapsed = time.perf_counter() - step_start
                sleep_s = max(raw_env.cfg.control_dt - elapsed, 0.0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            elif args.render_human:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if expert is not None:
            expert.close()
        env.close()

    if all_demos:
        output_path = Path(args.output)
        temp_output_path = output_path.with_name(f".{output_path.name}.tmp")
        with temp_output_path.open("wb") as file:
            pkl.dump(all_demos, file)
        os.replace(temp_output_path, output_path)
        print(
            f"\nSaved {demos_saved} insertion demos "
            f"({len(all_demos)} transitions) to {args.output}"
        )
    else:
        print("\nNo demos saved.")


if __name__ == "__main__":
    main()
