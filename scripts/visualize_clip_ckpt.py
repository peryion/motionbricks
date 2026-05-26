#!/usr/bin/env python3
"""Visualize MotionBricks clip-cache ckpt files that contain mujoco_qpos."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch


def load_ckpt(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    required = {"mujoco_qpos", "num_frames_per_clip"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
    return data


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

    def is_pressed(self, key: str) -> bool:
        return key in self.pressed


def disable_mujoco_keyboard_shortcuts(keys: str = "1234567890 np,.") -> None:
    try:
        from Xlib import X
        from Xlib import display as xdisplay
    except Exception:
        return

    try:
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
            for ch in keys:
                if ch == " ":
                    keysym = 32
                else:
                    keysym = ord(ch.upper())
                keycode = xdpy.keysym_to_keycode(keysym)
                mj_win.grab_key(keycode, X.AnyModifier, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.sync()
    except Exception as exc:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {exc}")


def draw_status(viewer, clip_idx: int, frame_idx: int, num_frames: int, paused: bool) -> None:
    if viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
        return
    pos = np.array([0.0, 0.0, 1.35], dtype=np.float64)
    color = np.array([1.0, 0.2, 0.1, 0.9], dtype=np.float32) if paused else np.array([0.1, 0.75, 0.25, 0.9], dtype=np.float32)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.06, 0.0, 0.0], dtype=np.float64),
        pos,
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    viewer.user_scn.ngeom += 1
    print(f"\rclip={clip_idx} frame={frame_idx}/{num_frames - 1} paused={paused}", end="", flush=True)


def main(args: argparse.Namespace) -> None:
    data = load_ckpt(Path(args.ckpt))
    qpos = data["mujoco_qpos"].float().numpy()
    num_frames = data["num_frames_per_clip"].int().numpy()

    model = mujoco.MjModel.from_xml_path(args.humanoid_xml)
    mj_data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.fps

    keys = KeyboardState()
    clip_idx = int(np.clip(args.clip, 0, qpos.shape[0] - 1))
    frame_idx = 0
    paused = False
    prev_pressed = set()

    print(f"Loaded {args.ckpt}")
    for idx, n in enumerate(num_frames):
        print(f"clip {idx}: {n} frames")
    print("Controls: 1-9 select clip, space pause, n next clip, p previous clip, . next frame, , previous frame")

    with mujoco.viewer.launch_passive(model, mj_data) as viewer:
        disable_mujoco_keyboard_shortcuts()
        while viewer.is_running():
            pressed = keys.pressed.copy()

            for digit in "123456789":
                if digit in pressed and digit not in prev_pressed:
                    target_idx = int(digit) - 1
                    if target_idx < qpos.shape[0]:
                        clip_idx = target_idx
                        frame_idx = 0

            if "n" in pressed and "n" not in prev_pressed:
                clip_idx = (clip_idx + 1) % qpos.shape[0]
                frame_idx = 0
            if "p" in pressed and "p" not in prev_pressed:
                clip_idx = (clip_idx - 1) % qpos.shape[0]
                frame_idx = 0
            if "space" in pressed and "space" not in prev_pressed:
                paused = not paused
            if "." in pressed and "." not in prev_pressed:
                frame_idx = (frame_idx + 1) % int(num_frames[clip_idx])
            if "," in pressed and "," not in prev_pressed:
                frame_idx = (frame_idx - 1) % int(num_frames[clip_idx])
            prev_pressed = pressed

            mj_data.qpos[:] = qpos[clip_idx, frame_idx]
            mujoco.mj_forward(model, mj_data)
            viewer.user_scn.ngeom = 0
            draw_status(viewer, clip_idx, frame_idx, int(num_frames[clip_idx]), paused)
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()

            if not paused:
                frame_idx = (frame_idx + 1) % int(num_frames[clip_idx])
            time.sleep(max(0.0, (1.0 / args.fps) / max(args.speed, 1e-6)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(root / "out/G1-pingpong-modes-clip.ckpt"))
    parser.add_argument("--humanoid_xml", default=str(root / "assets/skeletons/g1/scene_29dof.xml"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--clip", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
