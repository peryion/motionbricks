#!/usr/bin/env python3
"""Visualize converted MotionBricks motion-feature datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.motion_feature_dataset import MotionFeatureDataset
from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep


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


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def motion_to_qpos(sample: dict, motion_rep, converter, device: str) -> np.ndarray:
    motion = sample["motion"].float()[None].to(device)
    with torch.no_grad():
        qpos = converter.convert_motion_features_to_mujoco_qpos(
            motion,
            motion_rep,
            is_normalized=True,
            root_quat_w_first=True,
        )
    return qpos[0].detach().cpu().numpy()


def draw_status(viewer, clip_idx: int, frame_idx: int, num_frames: int, paused: bool) -> None:
    if viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
        return
    color = np.array([1.0, 0.2, 0.1, 0.9], dtype=np.float32) if paused else \
        np.array([0.1, 0.75, 0.25, 0.9], dtype=np.float32)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.06, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.35], dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    viewer.user_scn.ngeom += 1
    print(f"\rclip={clip_idx} frame={frame_idx}/{num_frames - 1} paused={paused}", end="", flush=True)


def main(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(_resolve(root, args.hparams))
    motion_rep = load_motion_rep(conf).to(args.device)
    converter = get_mujoco_converter(motion_rep, str(_resolve(root, args.skeleton_xml))).to(args.device)
    dataset = MotionFeatureDataset(_resolve(root, args.dataset), min_frames=1)

    model = mujoco.MjModel.from_xml_path(str(_resolve(root, args.humanoid_xml)))
    mj_data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.fps

    keys = KeyboardState()
    clip_idx = int(np.clip(args.clip, 0, len(dataset) - 1))
    frame_idx = 0
    paused = False
    prev_pressed = set()
    qpos = motion_to_qpos(dataset[clip_idx], motion_rep, converter, args.device)

    print(f"Loaded {args.dataset}: {len(dataset)} clips")
    print(f"Current key: {dataset[clip_idx]['keyid']}")
    print("Controls: 1-9 select clip, n next, p prev, space pause, . next frame, , prev frame")

    with mujoco.viewer.launch_passive(model, mj_data) as viewer:
        while viewer.is_running():
            pressed = keys.pressed.copy()

            new_clip_idx = clip_idx
            for digit in "123456789":
                if digit in pressed and digit not in prev_pressed:
                    target = int(digit) - 1
                    if target < len(dataset):
                        new_clip_idx = target
            if "n" in pressed and "n" not in prev_pressed:
                new_clip_idx = (clip_idx + 1) % len(dataset)
            if "p" in pressed and "p" not in prev_pressed:
                new_clip_idx = (clip_idx - 1) % len(dataset)
            if new_clip_idx != clip_idx:
                clip_idx = new_clip_idx
                frame_idx = 0
                qpos = motion_to_qpos(dataset[clip_idx], motion_rep, converter, args.device)
                print(f"\nCurrent key: {dataset[clip_idx]['keyid']}")

            if "space" in pressed and "space" not in prev_pressed:
                paused = not paused
            if "." in pressed and "." not in prev_pressed:
                frame_idx = (frame_idx + 1) % qpos.shape[0]
            if "," in pressed and "," not in prev_pressed:
                frame_idx = (frame_idx - 1) % qpos.shape[0]
            prev_pressed = pressed

            mj_data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(model, mj_data)
            viewer.user_scn.ngeom = 0
            draw_status(viewer, clip_idx, frame_idx, qpos.shape[0], paused)
            viewer.cam.lookat[:] = mj_data.qpos[:3]
            viewer.sync()

            if not paused:
                frame_idx = (frame_idx + 1) % qpos.shape[0]
            time.sleep(max(0.0, (1.0 / args.fps) / max(args.speed, 1e-6)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(root / "datasets/seed_g1_motion_features/train.pt"))
    parser.add_argument("--hparams", default=str(root / "out/motionbricks_pose/version_1/hparams.yaml"))
    parser.add_argument("--skeleton_xml", default=str(root / "assets/skeletons/g1/g1.xml"))
    parser.add_argument("--humanoid_xml", default=str(root / "assets/skeletons/g1/scene_29dof.xml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--clip", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
