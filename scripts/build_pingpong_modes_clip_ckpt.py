#!/usr/bin/env python3
"""Build a three-mode ping-pong clip cache from IsaacLab G1 npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as t
from omegaconf import OmegaConf

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep


MUJOCO_TO_ISAACLAB = np.array(
    [
        0, 6, 12,
        1, 7, 13,
        2, 8, 14,
        3, 9, 15, 22,
        4, 10, 16, 23,
        5, 11, 17, 24,
        18, 25,
        19, 26,
        20, 27,
        21, 28,
    ],
    dtype=np.int64,
)
ISAACLAB_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAACLAB)


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_npz_qpos(path: Path, target_fps: int, root_body_index: int) -> np.ndarray:
    data = np.load(path)
    source_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    root_pos = data["body_pos_w"][:, root_body_index, :]
    root_quat = data["body_quat_w"][:, root_body_index, :]
    joint_pos = data["joint_pos"][:, ISAACLAB_TO_MUJOCO]
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1).astype(np.float32)

    num_target_frames = max(5, int(np.floor(qpos.shape[0] * target_fps / source_fps)))
    frame_idx = np.round(np.arange(num_target_frames) * source_fps / target_fps).astype(np.int64)
    return qpos[np.clip(frame_idx, 0, qpos.shape[0] - 1)]


def _qpos_to_cache_tensors(converter, qpos: np.ndarray, motion_feature_dim: int) -> dict:
    qpos_tensor = t.from_numpy(qpos).float()[None]
    joint_pos, joint_rot = converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
    joint_pos = joint_pos[0].detach().cpu()
    joint_rot = joint_rot[0].detach().cpu()

    root_pos = joint_pos[:, 0] * t.tensor([[1.0, 0.0, 1.0]])
    root_relative_joint_pos = joint_pos - root_pos[:, None, :]
    root_dir = t.matmul(joint_rot[:, 0], t.tensor([0.0, 0.0, 1.0]).view(1, 3, 1)).view(-1, 3)
    root_dir = root_dir * t.tensor([1.0, 0.0, 1.0]).view(1, 3)
    root_dir = root_dir / (root_dir.norm(dim=1, keepdim=True) + 1e-5)

    return {
        "global_root_positions": root_pos,
        "global_joint_positions": root_relative_joint_pos,
        "global_joint_rotations": joint_rot,
        "global_headings": t.atan2(root_dir[:, 0], root_dir[:, 2]),
        "motion_feature": t.zeros((qpos.shape[0], motion_feature_dim), dtype=t.float32),
        "mujoco_qpos": t.from_numpy(qpos).float(),
        "num_frames": qpos.shape[0],
    }


def build(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(_resolve(root, args.hparams))
    motion_rep = load_motion_rep(conf)
    converter = get_mujoco_converter(motion_rep, str(_resolve(root, args.skeleton_xml)))
    target_fps = args.target_fps or motion_rep.fps
    motion_feature_dim = len(motion_rep.indices["all"])

    specs = [
        ("frozen_default", args.frozen),
        ("pingpong_forehand", args.forehand),
        ("pingpong_backhand", args.backhand),
    ]
    clips = []
    for name, path in specs:
        qpos = _load_npz_qpos(_resolve(root, path), target_fps, args.root_body_index)
        clips.append((name, _qpos_to_cache_tensors(converter, qpos, motion_feature_dim)))

    max_frames = max(int(data["num_frames"]) for _, data in clips)
    num_joints = clips[0][1]["global_joint_positions"].shape[1]
    shapes = {
        "global_root_positions": (3,),
        "global_joint_positions": (num_joints, 3),
        "global_joint_rotations": (num_joints, 3, 3),
        "global_headings": (),
        "motion_feature": (motion_feature_dim,),
        "mujoco_qpos": (36,),
    }
    state = {key: t.zeros((len(clips), max_frames, *shape), dtype=t.float32) for key, shape in shapes.items()}
    state["num_frames_per_clip"] = t.zeros((len(clips),), dtype=t.int32)

    for idx, (name, data) in enumerate(clips):
        n = int(data["num_frames"])
        state["num_frames_per_clip"][idx] = n
        for key in shapes:
            state[key][idx, :n] = data[key]
        print(f"{idx}: {name} frames={n}")

    output = _resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    t.save(state, output)
    print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="data/frozen_default_pose.npz")
    parser.add_argument("--forehand", default="data/forehand_no_yaw_float_z.npz")
    parser.add_argument("--backhand", default="data/backhand_no_yaw_float_z.npz")
    parser.add_argument("--output", default="out/G1-pingpong-modes-clip.ckpt")
    parser.add_argument("--hparams", default="out/motionbricks_pose/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", default="assets/skeletons/g1/g1.xml")
    parser.add_argument("--target_fps", type=int, default=None)
    parser.add_argument("--root_body_index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
