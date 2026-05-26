#!/usr/bin/env python3
"""Build a three-clip ping-pong cache from IsaacLab G1 npz files.

The output has the same tensor keys as out/G1-clip.ckpt, but contains only:
pingpong_forehand, pingpong_backhand, and frozen_default, in that order.
"""

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

DEFAULT_CLIPS = (
    ("pingpong_forehand", "data/forehand_no_yaw_float_z.npz"),
    ("pingpong_backhand", "data/backhand_no_yaw_float_z.npz"),
    ("frozen_default", "data/frozen_default_pose.npz"),
)


def _resolve(root: Path, path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else root / path_obj


def _load_isaaclab_npz_as_mujoco_qpos(npz_path: Path, target_fps: int, root_body_index: int) -> np.ndarray:
    data = np.load(npz_path)
    source_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    root_pos = data["body_pos_w"][:, root_body_index, :]
    root_quat = data["body_quat_w"][:, root_body_index, :]
    joint_pos = data["joint_pos"][:, ISAACLAB_TO_MUJOCO]
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1).astype(np.float32)

    num_target_frames = max(5, int(np.floor(qpos.shape[0] * target_fps / source_fps)))
    frame_idx = np.round(np.arange(num_target_frames) * source_fps / target_fps).astype(np.int64)
    frame_idx = np.clip(frame_idx, 0, qpos.shape[0] - 1)
    return qpos[frame_idx]


def _qpos_to_clip_tensors(converter, qpos: np.ndarray, motion_feature_dim: int) -> dict[str, t.Tensor | int]:
    qpos_tensor = t.from_numpy(qpos).float()[None]
    global_joint_positions, global_joint_rotations = converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)

    global_joint_positions = global_joint_positions[0].detach().cpu()
    global_joint_rotations = global_joint_rotations[0].detach().cpu()
    global_root_positions = global_joint_positions[:, 0] * t.tensor([[1.0, 0.0, 1.0]])
    root_relative_positions = global_joint_positions - global_root_positions[:, None, :]

    root_direction = t.matmul(
        global_joint_rotations[:, 0, :, :],
        t.tensor([0.0, 0.0, 1.0]).view([1, -1, 1]),
    ).view([-1, 3]) * t.tensor([1.0, 0.0, 1.0]).view([1, -1])
    root_direction = root_direction / (root_direction.norm(dim=1, keepdim=True) + 1e-5)
    global_headings = t.atan2(root_direction[:, 0], root_direction[:, 2])

    return {
        "global_root_positions": global_root_positions,
        "global_joint_positions": root_relative_positions,
        "global_joint_rotations": global_joint_rotations,
        "global_headings": global_headings,
        "motion_feature": t.zeros((qpos.shape[0], motion_feature_dim), dtype=t.float32),
        "mujoco_qpos": t.from_numpy(qpos).float(),
        "num_frames": qpos.shape[0],
    }


def build_ckpt(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    hparams = _resolve(root, args.hparams)
    skeleton_xml = _resolve(root, args.skeleton_xml)
    output = _resolve(root, args.output)

    conf = OmegaConf.load(hparams)
    motion_rep = load_motion_rep(conf)
    target_fps = args.target_fps or motion_rep.fps
    converter = get_mujoco_converter(motion_rep, str(skeleton_xml))
    motion_feature_dim = len(motion_rep.indices["all"])

    clip_specs = [
        ("pingpong_forehand", args.forehand),
        ("pingpong_backhand", args.backhand),
        ("frozen_default", args.frozen),
    ]
    clip_data = []
    for name, path in clip_specs:
        qpos = _load_isaaclab_npz_as_mujoco_qpos(_resolve(root, path), target_fps, args.root_body_index)
        tensors = _qpos_to_clip_tensors(converter, qpos, motion_feature_dim)
        clip_data.append((name, tensors))

    max_num_frames = max(int(data["num_frames"]) for _, data in clip_data)
    num_joints = clip_data[0][1]["global_joint_positions"].shape[1]
    state_dict = {}
    shapes = {
        "global_root_positions": (3,),
        "global_joint_positions": (num_joints, 3),
        "global_joint_rotations": (num_joints, 3, 3),
        "global_headings": (),
        "motion_feature": (motion_feature_dim,),
        "mujoco_qpos": (36,),
    }

    for key, shape in shapes.items():
        state_dict[key] = t.zeros((len(clip_data), max_num_frames, *shape), dtype=t.float32)

    num_frames_per_clip = t.zeros((len(clip_data),), dtype=t.int32)
    for clip_idx, (_, data) in enumerate(clip_data):
        clip_length = int(data["num_frames"])
        num_frames_per_clip[clip_idx] = clip_length
        for key in shapes:
            state_dict[key][clip_idx, :clip_length] = data[key]
    state_dict["num_frames_per_clip"] = num_frames_per_clip

    output.parent.mkdir(parents=True, exist_ok=True)
    t.save(state_dict, output)
    print(f"Saved {output}")
    for idx, (name, _) in enumerate(clip_data):
        print(f"{idx}: {name} frames={int(num_frames_per_clip[idx])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forehand", default=DEFAULT_CLIPS[0][1])
    parser.add_argument("--backhand", default=DEFAULT_CLIPS[1][1])
    parser.add_argument("--frozen", default=DEFAULT_CLIPS[2][1])
    parser.add_argument("--output", default="out/G1-pingpong-clip.ckpt")
    parser.add_argument("--hparams", default="out/motionbricks_pose/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", default="assets/skeletons/g1/g1.xml")
    parser.add_argument("--target_fps", type=int, default=None)
    parser.add_argument("--root_body_index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    build_ckpt(parse_args())
