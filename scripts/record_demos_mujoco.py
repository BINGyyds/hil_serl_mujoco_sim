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

from serl_robot_infra.common.async_camera_window import AsyncCameraWindow
from serl_robot_infra.common.config import VisualizationConfig
from serl_robot_infra.configs.insertion_config import make_insertion_config
from serl_robot_infra.envs.franka_insertion_env import FrankaInsertionEnv
from serl_robot_infra.envs.wrappers import MujocoSERLObsWrapper
from serl_robot_infra.spacemouse.spacemouse_expert import SpaceMouseExpert


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
        "--show-cameras",
        action="store_true",
        help="打开末端相机观察窗口；画面只用于辅助录制，不写入 demo。",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=20,
        help="--show-cameras 相机窗口刷新率。",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=320,
        help="--show-cameras 单路相机宽度。",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=240,
        help="--show-cameras 单路相机高度。",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按 env.control_dt 近似做实时节奏推进，主要用于配合 --render-human 观察。",
    )
    parser.add_argument(
        "--random-reset",
        action="store_true",
        help="恢复训练用 reset 随机扰动；默认关闭，使录制起点与 debug teleop home 一致。",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=100.0,
        help="录制时 SpaceMouse action 更新频率。默认 100Hz，与训练 actor/eval 默认一致。",
    )
    parser.add_argument(
        "--sim-timestep",
        type=float,
        default=0.001,
        help="录制时 MuJoCo 物理步长。默认 0.001s，与 debug teleop 一致。",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="单条 demo 最多允许的环境步数。默认 1000 步约 10 秒。",
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
    if args.control_hz <= 0.0:
        raise ValueError("--control-hz 必须大于 0。")
    if args.sim_timestep <= 0.0:
        raise ValueError("--sim-timestep 必须大于 0。")
    control_dt = 1.0 / args.control_hz
    n_substeps = max(1, int(round(control_dt / args.sim_timestep)))
    control_dt = n_substeps * args.sim_timestep

    config_overrides = dict(
        max_episode_steps=args.max_episode_steps,
        sim_timestep=args.sim_timestep,
        n_substeps=n_substeps,
        control_dt=control_dt,
    )
    if not args.random_reset:
        config_overrides.update(
            reset_pos_noise_low=np.zeros(3, dtype=np.float64),
            reset_pos_noise_high=np.zeros(3, dtype=np.float64),
            reset_yaw_range=0.0,
        )
    raw_env = FrankaInsertionEnv(
        config=make_insertion_config(**config_overrides),
        render_mode=render_mode,
    )
    env = MujocoSERLObsWrapper(raw_env, proprio_keys=proprio_keys)
    expert = make_expert()
    camera_window: Optional[AsyncCameraWindow] = None
    if args.show_cameras:
        camera_config = VisualizationConfig()
        camera_window = AsyncCameraWindow(
            model=raw_env.model,
            data=raw_env.data,
            camera_names=camera_config.teleop_camera_names,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            window_name=camera_config.teleop_camera_window_name,
        )

    all_demos: list = []
    episode_transitions: list = []
    demos_saved = 0

    print(f"Recording {args.num_demos} insertion demonstrations.")
    print("  SpaceMouse controls the tool tip (6 DOF).")
    print("  RIGHT button = save successful episode | LEFT button = discard/reset")
    print(f"  Max episode steps: {args.max_episode_steps}")
    print(f"  Action scale: {args.action_scale}")
    print(
        f"  Control rate: {1.0 / raw_env.cfg.control_dt:.1f} Hz "
        f"(dt={raw_env.cfg.control_dt:.4f}s, "
        f"sim_timestep={raw_env.cfg.sim_timestep:.4f}s, "
        f"n_substeps={raw_env.cfg.n_substeps})"
    )
    print(f"  Random reset: {args.random_reset}")
    if args.render_human:
        print("  Human viewer enabled.")
    else:
        print("  Running headless.")
    if args.show_cameras:
        print("  Teleop camera window enabled.")
    print("-" * 60)

    obs, _ = env.reset()
    maybe_render(env, args.render_human)
    if camera_window is not None:
        camera_window.reset_timing()
        camera_window.render()
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
            if camera_window is not None:
                camera_window.render()

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
                    f"rot_error={info.get('rot_error', 0.0):.4f}, "
                    f"linear_speed={info.get('tcp_linear_speed', 0.0):.4f}, "
                    f"angular_speed={info.get('tcp_angular_speed', 0.0):.4f})"
                )
                episode_transitions = []
                obs, _ = env.reset()
                maybe_render(env, args.render_human)
                if camera_window is not None:
                    camera_window.reset_timing()
                    camera_window.render()
                time.sleep(0.3)
            elif should_discard:
                print(
                    f"  [DISCARDED] Episode ({len(episode_transitions)} steps, "
                    f"success={info.get('success', False)}, "
                    f"xy_error={info.get('xy_error', 0.0):.4f}, "
                    f"z_gap={info.get('z_gap', 0.0):.4f}, "
                    f"rot_error={info.get('rot_error', 0.0):.4f}, "
                    f"linear_speed={info.get('tcp_linear_speed', 0.0):.4f}, "
                    f"angular_speed={info.get('tcp_angular_speed', 0.0):.4f})"
                )
                episode_transitions = []
                obs, _ = env.reset()
                maybe_render(env, args.render_human)
                if camera_window is not None:
                    camera_window.reset_timing()
                    camera_window.render()
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
        if camera_window is not None:
            camera_window.close()
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
