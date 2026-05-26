#!/usr/bin/env python3
"""Ping-pong target controller demo for the G1 humanoid.

Press N in the viewer to cycle to the next ping-pong target.

Key differences from interactive_demo_g1.py:
- Controller is always PingPongTargetController (no --controller flag).
- All pingpong_* parameters are exposed as CLI arguments.
"""

import argparse
import time
import platform

import mujoco
import mujoco.viewer
import numpy as np
import torch as t

from motionbricks.motion_backbone.demo.pingpong_agent import PingPongDemo


def _disable_mujoco_keyboard_shortcuts(controller_keys: str = "wasdrtfgeqzxcvb") -> None:
    if platform.system() != "Linux":
        return
    try:
        from Xlib import display as xdisplay, X
        xdpy = xdisplay.Display()
        root = xdpy.screen().root

        def find_window_by_name(win, name_substr):
            try:
                name = win.get_wm_name()
                if name and name_substr in name:
                    return win
            except Exception:
                pass
            for child in win.query_tree().children:
                result = find_window_by_name(child, name_substr)
                if result:
                    return result
            return None

        time.sleep(0.5)
        mj_win = find_window_by_name(root, "MuJoCo")
        if mj_win:
            for ch in controller_keys:
                keycode = xdpy.keysym_to_keycode(ord(ch) - 32)
                mj_win.grab_key(keycode, X.AnyModifier, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.sync()
    except Exception as e:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {e}")


def _parse_float_tuple(text: str, expected_len: int):
    values = tuple(float(v) for v in text.split(","))
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"expected {expected_len} comma-separated values, got: {text!r}"
        )
    return values


def main(args) -> None:
    demo_agent = PingPongDemo(args)

    for run_idx in range(args.num_runs):
        print(f"Running iteration {run_idx + 1} / {args.num_runs}")
        random_seed = args.random_seed * (run_idx + 2333) * 2333 % (2 ** 32 - 1)
        np.random.seed(random_seed)
        t.manual_seed(random_seed)
        demo_agent.full_agent.reset()

        steps = 0

        if args.has_viewer:
            with mujoco.viewer.launch_passive(demo_agent.mj_model, demo_agent.mj_data) as viewer:
                _disable_mujoco_keyboard_shortcuts()
                while viewer.is_running() and steps < args.max_steps:
                    force_idle = steps + 100 > args.max_steps
                    steps += 1
                    viewer.user_scn.ngeom = 0
                    step_start = time.time()

                    qpos = demo_agent.full_agent.get_next_frame()
                    context_motion_features = demo_agent.full_agent.get_context_motion_features()
                    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                    demo_agent.mj_data.qpos[:] = qpos

                    control_signals = demo_agent.controller.generate_control_signals(
                        viewer, demo_agent.mj_model, demo_agent.mj_data,
                        visualize=True,
                        control_info={"force_idle": force_idle,
                                      "allowed_mode": getattr(args, "allowed_mode", None)},
                    )

                    if args.use_qpos:
                        control_signals["context_mujoco_qpos"] = context_mujoco_qpos
                    else:
                        control_signals["context_motion_features"] = context_motion_features

                    force_generation = bool(control_signals.pop("force_generation", False))
                    skip_generation = bool(control_signals.pop("skip_generation", False))

                    if not skip_generation:
                        with t.no_grad():
                            demo_agent.full_agent.generate_new_frames(
                                control_signals,
                                demo_agent.controller.get_controller_dt() * args.generate_dt,
                                force_generation=force_generation,
                            )

                    mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
                    viewer.cam.lookat[:] = demo_agent.controller.get_prev_qpos()[:, :3].mean(axis=0)
                    viewer.sync()

                    time_until_next_step = demo_agent.mj_model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while steps < args.max_steps:
                steps += 1
                force_idle = steps + 100 > args.max_steps
                qpos = demo_agent.full_agent.get_next_frame()
                context_motion_features = demo_agent.full_agent.get_context_motion_features()
                context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                demo_agent.mj_data.qpos[:] = qpos

                control_signals = demo_agent.controller.generate_control_signals(
                    None, demo_agent.mj_model, demo_agent.mj_data,
                    visualize=False,
                    control_info={"force_idle": force_idle,
                                  "allowed_mode": getattr(args, "allowed_mode", None)},
                )

                if args.use_qpos:
                    control_signals["context_mujoco_qpos"] = context_mujoco_qpos
                else:
                    control_signals["context_motion_features"] = context_motion_features

                force_generation = bool(control_signals.pop("force_generation", False))
                skip_generation = bool(control_signals.pop("skip_generation", False))

                if not skip_generation:
                    with t.no_grad():
                        demo_agent.full_agent.generate_new_frames(
                            control_signals,
                            demo_agent.controller.get_controller_dt() * args.generate_dt,
                            force_generation=force_generation,
                        )

                mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ping-pong target demo for the G1 humanoid. Press N to cycle targets."
    )

    # paths
    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    # viewer / rendering
    parser.add_argument("--has_viewer", type=int, default=1)
    parser.add_argument("--lookat_movement_direction", type=int, default=0)

    # motion generation
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    # pingpong target constraints
    parser.add_argument(
        "--pingpong_bank_source", "--pingpong-bank-source",
        type=str, default="existing", choices=["existing", "procedural"],
        help="'existing': cycle through preset clip bank (forehand/backhand). "
             "'procedural': sample targets from x/y/z ranges.",
    )
    parser.add_argument(
        "--pingpong_target_x_range", "--pingpong-target-x-range",
        type=lambda s: _parse_float_tuple(s, 2), default=(-0.45, -0.18),
        metavar="MIN,MAX",
        help="Right-hand target x range (lateral) in root-relative MotionBricks coords. "
             "Only used when --pingpong_bank_source=procedural.",
    )
    parser.add_argument(
        "--pingpong_target_y_range", "--pingpong-target-y-range",
        type=lambda s: _parse_float_tuple(s, 2), default=(0.85, 1.15),
        metavar="MIN,MAX",
        help="Right-hand target y range (height). Only for procedural bank.",
    )
    parser.add_argument(
        "--pingpong_target_z_range", "--pingpong-target-z-range",
        type=lambda s: _parse_float_tuple(s, 2), default=(0.35, 0.75),
        metavar="MIN,MAX",
        help="Right-hand target z range (forward). Only for procedural bank.",
    )
    parser.add_argument(
        "--pingpong_swing_velocity", "--pingpong-swing-velocity",
        type=lambda s: _parse_float_tuple(s, 3), default=(-0.20, 0.00, 1.20),
        metavar="X,Y,Z",
        help="Swing velocity vector that defines the paddle facing direction. Only for procedural bank.",
    )
    parser.add_argument(
        "--pingpong_mode", "--pingpong-mode",
        type=str, default="none",
        help="Motion style/primitive for targets (e.g. 'walk_boxing'). "
             "'none' disables primitive sampling. Only for procedural bank.",
    )
    parser.add_argument(
        "--pingpong_root_step_scale", "--pingpong-root-step-scale",
        type=float, default=1.0,
        help="Scale multiplier for root offset steps.",
    )
    parser.add_argument(
        "--pingpong_root_lateral_distance", "--pingpong-root-lateral-distance",
        type=float, default=0.5,
        help="Metres from anchor to each left/right root target.",
    )
    parser.add_argument(
        "--pingpong_target_tokens", "--pingpong-target-tokens",
        type=int, default=6,
        help="Prediction horizon in tokens when a target is active (1 token = 4 frames).",
    )
    parser.add_argument(
        "--pingpong_bank_json", "--pingpong-bank-json",
        type=str, default=None,
        metavar="PATH",
        help="Path to a JSON file defining a custom bank (overrides --pingpong_bank_source). "
             "See scripts/example_pingpong_bank.json for the format.",
    )

    # run
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)

    # model
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    args = parser.parse_args()
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(v) for v in args.speed_scale.split(",")]

    main(args)
