#!/usr/bin/env python3
"""Stream MotionBricks G1 motion to SONIC's ZMQ motion-tracking input."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch as t
import zmq


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTIONBRICKS_ROOT = REPO_ROOT / "motionbricks"
for path in (str(REPO_ROOT), str(MOTIONBRICKS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
from motionbricks.motion_backbone.demo.utils import navigation_demo


# SONIC expects Protocol v1 joint arrays in IsaacLab order. MotionBricks qpos
# follows the G1 MuJoCo XML order, so index the MuJoCo vector with this map.
MUJOCO_TO_ISAACLAB = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)


def _disable_mujoco_keyboard_shortcuts(controller_keys: str = "wasdrtfgeqzxcvb") -> None:
    if platform.system() != "Linux":
        return
    try:
        from Xlib import X
        from Xlib import display as xdisplay

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
            for char in controller_keys:
                keycode = xdpy.keysym_to_keycode(ord(char) - 32)
                mj_win.grab_key(keycode, X.AnyModifier, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.sync()
    except Exception as exc:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {exc}")


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, t.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_float_tuple(text: str, expected_len: int) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(","))
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"expected {expected_len} comma-separated values, got {text}")
    return values


def _current_qpos_window(full_agent, horizon: int) -> np.ndarray:
    qpos_buffer = _to_numpy(full_agent.frames["mujoco_qpos"][0])
    start = max(int(full_agent._current_frame_idx) - 1, 0)
    indices = np.clip(np.arange(start, start + horizon), 0, qpos_buffer.shape[0] - 1)
    return qpos_buffer[indices].astype(np.float32, copy=False)


def _normalize_quaternions_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    bad = norm.squeeze(-1) < 1e-6
    norm[bad] = 1.0
    quat /= norm
    quat[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat


def _build_sonic_pose_packet(qpos_window: np.ndarray, frame_start: int, dt: float) -> dict[str, np.ndarray]:
    if qpos_window.ndim == 1:
        qpos_window = qpos_window[None]
    if qpos_window.shape[1] < 36:
        raise ValueError(f"Expected G1 qpos with at least 36 values, got {qpos_window.shape}")

    body_quat = _normalize_quaternions_wxyz(qpos_window[:, 3:7])
    joint_pos = qpos_window[:, 7:36][:, MUJOCO_TO_ISAACLAB].astype(np.float32, copy=False)

    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    if len(joint_pos) > 1:
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt
        joint_vel[-1] = joint_vel[-2]

    frame_index = np.arange(frame_start, frame_start + len(qpos_window), dtype=np.int64)
    return {
        "body_quat": body_quat,
        "frame_index": frame_index,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "catch_up": np.array([True], dtype=np.bool_),
    }


def _make_socket(args) -> tuple[zmq.Context, zmq.Socket]:
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    if args.zmq_conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    if args.zmq_connect:
        socket.connect(endpoint)
        mode = "connect"
    else:
        socket.bind(endpoint)
        mode = "bind"
    print(f"SONIC ZMQ publisher {mode}: {endpoint}, topic={args.zmq_topic}")
    if args.zmq_warmup_sec > 0:
        time.sleep(args.zmq_warmup_sec)
    return context, socket


def _step_motionbricks(demo_agent, args, viewer, steps: int) -> np.ndarray:
    force_idle = steps + 100 > args.max_steps
    qpos = demo_agent.full_agent.get_next_frame()
    context_motion_features = demo_agent.full_agent.get_context_motion_features()
    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
    qpos_window = _current_qpos_window(demo_agent.full_agent, args.window_frames)
    demo_agent.mj_data.qpos[:] = qpos

    control_signals = demo_agent.controller.generate_control_signals(
        viewer,
        demo_agent.mj_model,
        demo_agent.mj_data,
        visualize=viewer is not None,
        control_info={"force_idle": force_idle, "allowed_mode": getattr(args, "allowed_mode", None)},
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
    return qpos_window


def _run_loop(demo_agent, args, socket, viewer=None) -> None:
    frame_start = 0
    dt = float(demo_agent.mj_model.opt.timestep)
    print(
        "Streaming MotionBricks qpos as SONIC Protocol v1 "
        f"({args.window_frames} frames/message, dt={dt:.4f}s)."
    )

    steps = 0
    while steps < args.max_steps and (viewer is None or viewer.is_running()):
        step_begin = time.time()
        steps += 1

        if viewer is not None:
            viewer.user_scn.ngeom = 0

        qpos_window = _step_motionbricks(demo_agent, args, viewer, steps)
        pose_data = _build_sonic_pose_packet(qpos_window, frame_start, dt)
        socket.send(pack_pose_message(pose_data, topic=args.zmq_topic, version=1))

        if viewer is not None:
            viewer.cam.lookat[:] = demo_agent.controller.get_prev_qpos()[:, :3].mean(axis=0)
            viewer.sync()

        if args.print_every > 0 and steps % args.print_every == 0:
            print(
                f"sent step={steps} frame={frame_start} "
                f"joint_pos={pose_data['joint_pos'].shape}"
            )

        frame_start += 1
        sleep_time = max(0.0, dt / args.publish_speed - (time.time() - step_begin))
        if sleep_time > 0:
            time.sleep(sleep_time)


def main(args) -> None:
    demo_agent = navigation_demo(args)
    context, socket = _make_socket(args)
    try:
        for run_idx in range(args.num_runs):
            print(f"Running iteration {run_idx + 1} / {args.num_runs}")
            random_seed = args.random_seed * (run_idx + 2333) * 2333 % (2**32 - 1)
            np.random.seed(random_seed)
            t.manual_seed(random_seed)
            demo_agent.full_agent.reset()

            if args.has_viewer:
                with mujoco.viewer.launch_passive(demo_agent.mj_model, demo_agent.mj_data) as viewer:
                    _disable_mujoco_keyboard_shortcuts()
                    _run_loop(demo_agent, args, socket, viewer)
            else:
                _run_loop(demo_agent, args, socket, None)
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bridge MotionBricks G1 motion generation into SONIC ZMQ control."
    )

    parser.add_argument(
        "--humanoid_scene_xml",
        type=str,
        default=str(MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "scene_29dof.xml"),
    )
    parser.add_argument(
        "--skeleton_xml",
        type=str,
        default=str(MOTIONBRICKS_ROOT / "assets" / "skeletons" / "g1" / "g1.xml"),
    )
    parser.add_argument("--result_dir", type=str, default=str(MOTIONBRICKS_ROOT / "out"))
    parser.add_argument("--data_root", type=str, default=str(MOTIONBRICKS_ROOT / "datasets"))
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    parser.add_argument("--clips_ckpt", type=str, default=str(MOTIONBRICKS_ROOT / "out" / "G1-clip.ckpt"))

    parser.add_argument("--controller", type=str, default="wasd", choices=["wasd", "random", "pingpong"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--has_viewer", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument(
        "--pingpong_target_x_range",
        "--pingpong-target-x-range",
        type=lambda text: _parse_float_tuple(text, 2),
        default=(-0.45, -0.18),
        help="Right-hand target root-relative x range in MotionBricks coordinates.",
    )
    parser.add_argument(
        "--pingpong_target_y_range",
        "--pingpong-target-y-range",
        type=lambda text: _parse_float_tuple(text, 2),
        default=(0.85, 1.15),
        help="Right-hand target root-relative y/up range in MotionBricks coordinates.",
    )
    parser.add_argument(
        "--pingpong_target_z_range",
        "--pingpong-target-z-range",
        type=lambda text: _parse_float_tuple(text, 2),
        default=(0.35, 0.75),
        help="Right-hand target root-relative z/forward range in MotionBricks coordinates.",
    )
    parser.add_argument(
        "--pingpong_swing_velocity",
        "--pingpong-swing-velocity",
        type=lambda text: _parse_float_tuple(text, 3),
        default=(-0.20, 0.00, 1.20),
        help="Right-hand target velocity encoded across the four target frames.",
    )
    parser.add_argument(
        "--pingpong_root_step_scale",
        "--pingpong-root-step-scale",
        type=float,
        default=1.0,
        help="Multiplier for pingpong root target offsets.",
    )
    parser.add_argument(
        "--pingpong_root_lateral_distance",
        "--pingpong-root-lateral-distance",
        type=float,
        default=0.5,
        help="Meters from the startup root anchor to each left/right pingpong root target.",
    )
    parser.add_argument(
        "--pingpong_bank_source",
        "--pingpong-bank-source",
        type=str,
        default="existing",
        choices=["existing", "procedural"],
        help="Use fixed frames from the existing clip bank, or the experimental procedural bank.",
    )
    parser.add_argument(
        "--pingpong_target_tokens",
        "--pingpong-target-tokens",
        type=int,
        default=6,
        help="One-shot target horizon in MotionBricks tokens. One token is 4 frames.",
    )
    parser.add_argument(
        "--pingpong_mode",
        "--pingpong-mode",
        type=str,
        default="none",
        help="Primitive/style mode for pingpong. Use 'none' to disable target primitive sampling.",
    )

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)

    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    parser.add_argument("--zmq_host", "--zmq-host", type=str, default="*")
    parser.add_argument("--zmq_port", "--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq_topic", "--zmq-topic", type=str, default="pose")
    parser.add_argument("--zmq_connect", "--zmq-connect", action="store_true")
    parser.add_argument("--zmq_conflate", "--zmq-conflate", action="store_true")
    parser.add_argument("--zmq_warmup_sec", "--zmq-warmup-sec", type=float, default=0.5)
    parser.add_argument("--window_frames", "--window-frames", type=int, default=30)
    parser.add_argument("--publish_speed", "--publish-speed", type=float, default=1.0)
    parser.add_argument("--print_every", "--print-every", type=int, default=60)

    args = parser.parse_args()
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(item) for item in args.speed_scale.split(",")]
    args.window_frames = max(1, args.window_frames)
    args.publish_speed = max(1e-6, args.publish_speed)

    main(args)
