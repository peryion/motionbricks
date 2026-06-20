#!/usr/bin/env python3
"""Visualize generated ping-pong reference-buffer npz files and command metadata."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pingpong_demo_utils import disable_mujoco_keyboard_shortcuts, draw_pingpong_table

MUJOCO_TO_ISAACLAB = np.array(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAACLAB)
class KeyboardState:
    def __init__(self):
        from pynput import keyboard

        self.pressed = set()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def _on_press(self, key):
        char = getattr(key, "char", None) or getattr(key, "name", None)
        if char:
            self.pressed.add(char.lower())

    def _on_release(self, key):
        char = getattr(key, "char", None) or getattr(key, "name", None)
        if char:
            self.pressed.discard(char.lower())


def _load_motion(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {
        "fps",
        "joint_pos",
        "body_pos_w",
        "body_quat_w",
        "command_start_frames",
        "hit_pos",
        "hit_vel",
        "root_target",
        "stroke_sign",
        "hit_time_s",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing reference command metadata: {missing}")
    return {key: data[key] for key in data.files}


def _qpos_from_isaaclab_motion(data: dict[str, np.ndarray]) -> np.ndarray:
    root_pos = data["body_pos_w"][:, 0, :3]
    root_quat = data["body_quat_w"][:, 0, :4]
    joint_pos_mujoco = data["joint_pos"][:, ISAACLAB_TO_MUJOCO]
    return np.concatenate([root_pos, root_quat, joint_pos_mujoco], axis=-1).astype(np.float32)


def _current_command_index(data: dict[str, np.ndarray], frame_idx: int) -> int:
    starts = data["command_start_frames"].astype(np.int64)
    return int(np.clip(np.searchsorted(starts, frame_idx, side="right") - 1, 0, len(starts) - 1))


def _add_geom(viewer, geom_type, pos, size, rgba, mat=None, label: str | None = None) -> None:
    if viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
        return
    if mat is None:
        mat = np.eye(3, dtype=np.float64)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    if label is not None:
        viewer.user_scn.geoms[viewer.user_scn.ngeom].label = label
    viewer.user_scn.ngeom += 1


def _draw_command(
    viewer,
    data: dict[str, np.ndarray],
    frame_idx: int,
    fps: float,
    clip_hit_countdown_min: float | None,
) -> None:
    cmd_idx = _current_command_index(data, frame_idx)
    start = int(data["command_start_frames"][cmd_idx])
    elapsed = (frame_idx - start) / fps
    raw_countdown = float(data["hit_time_s"][cmd_idx]) - elapsed
    if "time_to_hit_s" in data:
        countdown = float(data["time_to_hit_s"][frame_idx])
    else:
        countdown = raw_countdown
    if "time_to_hit_s" not in data and clip_hit_countdown_min is not None:
        countdown = max(float(clip_hit_countdown_min), countdown)
    hit_pos = data["hit_pos"][cmd_idx].astype(np.float64)
    hit_vel = data["hit_vel"][cmd_idx].astype(np.float64)
    root_target = data["root_target"][cmd_idx].astype(np.float64)
    stroke = "forehand" if float(data["stroke_sign"][cmd_idx]) > 0.0 else "backhand"

    _add_geom(viewer, mujoco.mjtGeom.mjGEOM_SPHERE, hit_pos, [0.045, 0.0, 0.0], [1.0, 0.15, 0.1, 0.75])
    _add_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        hit_pos + 0.22 * hit_vel,
        [0.03, 0.0, 0.0],
        [1.0, 0.85, 0.05, 0.7],
    )
    _add_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [root_target[0], root_target[1], 0.78],
        [0.055, 0.0, 0.0],
        [0.1, 0.65, 1.0, 0.45],
    )
    label = f"{stroke} cmd={cmd_idx} hit T={countdown:+.2f}s"
    _add_geom(
        viewer,
        mujoco.mjtGeom.mjGEOM_LABEL,
        hit_pos + np.array([0.0, 0.0, 0.18]),
        [0.02, 0.02, 0.02],
        [1.0, 1.0, 1.0, 1.0],
        label=label,
    )
    print(
        f"\rframe={frame_idx:04d}/{data['joint_pos'].shape[0] - 1:04d} "
        f"cmd={cmd_idx} {stroke} hit_countdown={countdown:+.2f}s raw={raw_countdown:+.2f}s "
        f"hit={np.round(hit_pos, 3)} vel={np.round(hit_vel, 3)}",
        end="",
        flush=True,
    )


def _discover_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.npz"))
    if not files:
        files = sorted(path.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files


def main(args: argparse.Namespace) -> None:
    files = _discover_npz(Path(args.path))
    model = mujoco.MjModel.from_xml_path(args.humanoid_xml)
    mj_data = mujoco.MjData(model)

    file_idx = int(np.clip(args.clip, 0, len(files) - 1))
    data = _load_motion(files[file_idx])
    qpos = _qpos_from_isaaclab_motion(data)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    model.opt.timestep = 1.0 / fps

    frame_idx = 0
    paused = False
    keys = KeyboardState()
    prev_pressed = set()

    print(f"Loaded {len(files)} reference files from {args.path}")
    print("Controls: space pause, n/p next/prev clip, ./, step frame, 1-9 select clip")

    with mujoco.viewer.launch_passive(model, mj_data) as viewer:
        disable_mujoco_keyboard_shortcuts("np .,123456789")
        while viewer.is_running():
            pressed = keys.pressed.copy()
            if "space" in pressed and "space" not in prev_pressed:
                paused = not paused
            if "n" in pressed and "n" not in prev_pressed:
                file_idx = (file_idx + 1) % len(files)
                data = _load_motion(files[file_idx])
                qpos = _qpos_from_isaaclab_motion(data)
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                frame_idx = 0
                print(f"\nLoaded {files[file_idx]}")
            if "p" in pressed and "p" not in prev_pressed:
                file_idx = (file_idx - 1) % len(files)
                data = _load_motion(files[file_idx])
                qpos = _qpos_from_isaaclab_motion(data)
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                frame_idx = 0
                print(f"\nLoaded {files[file_idx]}")
            for digit in "123456789":
                if digit in pressed and digit not in prev_pressed:
                    target = int(digit) - 1
                    if target < len(files):
                        file_idx = target
                        data = _load_motion(files[file_idx])
                        qpos = _qpos_from_isaaclab_motion(data)
                        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                        frame_idx = 0
                        print(f"\nLoaded {files[file_idx]}")
            if "." in pressed and "." not in prev_pressed:
                frame_idx = min(frame_idx + 1, qpos.shape[0] - 1)
            if "," in pressed and "," not in prev_pressed:
                frame_idx = max(frame_idx - 1, 0)
            prev_pressed = pressed

            mj_data.qpos[:] = qpos[frame_idx]
            if frame_idx == 459:
                print("debug frame")
            mujoco.mj_forward(model, mj_data)
            viewer.user_scn.ngeom = 0
            draw_pingpong_table(viewer, args.table_position, args.table_yaw)
            _draw_command(viewer, data, frame_idx, fps, args.clip_hit_countdown_min)
            if args.follow_robot:
                viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()

            if not paused:
                frame_idx = (frame_idx + 1) % qpos.shape[0]
            time.sleep(max(0.0, (1.0 / fps) / max(args.speed, 1e-6)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(root / "out/reference_ball/pingpong_fk/ready"))
    parser.add_argument("--humanoid_xml", default=str(root / "assets/skeletons/g1/g1_29dof_tt.xml"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--table_position", type=float, nargs=3, default=(1.87, 0.0, 0.74))
    parser.add_argument("--table_yaw", type=float, default=0.0)
    parser.add_argument("--follow_robot", action="store_true", help="Keep the camera lookat on the robot root.")
    parser.add_argument(
        "--clip_hit_countdown_min",
        type=float,
        default=-0.75,
        help="Clip the displayed hit countdown lower bound. Use none by passing a very negative value if needed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
