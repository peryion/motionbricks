#!/usr/bin/env python3
"""Convert GMR ping-pong G1 pkl motions to MotionBricks training features."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _resample_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-5:
        return np.arange(num_frames, dtype=np.int64)
    target_frames = max(1, int(np.floor(num_frames * target_fps / source_fps)))
    indices = np.round(np.arange(target_frames) * source_fps / target_fps).astype(np.int64)
    return np.clip(indices, 0, num_frames - 1)


def _load_gmr_qpos(path: Path, target_fps: int) -> tuple[np.ndarray, float]:
    with path.open("rb") as f:
        data = pickle.load(f)
    required = {"root_pos", "root_rot", "dof_pos"}
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof = np.asarray(data["dof_pos"], dtype=np.float32)
    source_fps = float(data.get("fps", target_fps))

    if root_pos.shape[0] != root_quat_xyzw.shape[0] or root_pos.shape[0] != dof.shape[0]:
        raise ValueError("root_pos, root_rot, and dof_pos have inconsistent lengths")
    if root_quat_xyzw.shape[-1] != 4 or dof.shape[-1] != 29:
        raise ValueError(f"Expected root_rot (*,4) and dof_pos (*,29), got {root_quat_xyzw.shape}, {dof.shape}")

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    root_quat_wxyz /= np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True).clip(min=1e-8)
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof], axis=-1).astype(np.float32)
    return qpos[_resample_indices(qpos.shape[0], source_fps, target_fps)], source_fps


def _qpos_to_motion_feature(converter, motion_rep, qpos: np.ndarray, device: str) -> torch.Tensor:
    with torch.no_grad():
        qpos_tensor = torch.from_numpy(qpos).float()[None].to(device)
        joint_pos, joint_rot = converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
        lengths = torch.tensor([qpos.shape[0]], device=device)
        features = motion_rep.dual_rep.global_motion_rep(
            {"posed_joints": joint_pos, "global_joint_rots": joint_rot},
            to_normalize=True,
            lengths=lengths,
        )
    return features[0].detach().cpu()


def _iter_chunks(qpos: np.ndarray, chunk_frames: int | None, overlap_frames: int) -> list[tuple[int, np.ndarray]]:
    if chunk_frames is None or chunk_frames <= 0 or qpos.shape[0] <= chunk_frames:
        return [(0, qpos)]
    step = max(1, chunk_frames - max(0, overlap_frames))
    chunks = []
    for start in range(0, qpos.shape[0] - chunk_frames + 1, step):
        chunks.append((start, qpos[start:start + chunk_frames]))
    if chunks and chunks[-1][0] + chunk_frames < qpos.shape[0]:
        chunks.append((qpos.shape[0] - chunk_frames, qpos[-chunk_frames:]))
    return chunks


def convert(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = _resolve(root, args.input)
    output = _resolve(root, args.output)
    conf = OmegaConf.load(_resolve(root, args.hparams))
    motion_rep = load_motion_rep(conf).to(args.device)
    converter = get_mujoco_converter(motion_rep, str(_resolve(root, args.skeleton_xml))).to(args.device)
    target_fps = int(args.target_fps or motion_rep.fps)
    min_frames = int(args.min_tokens * args.frames_per_token + 1)

    paths = [input_path] if input_path.is_file() else sorted(input_path.rglob(args.pattern))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    if not paths:
        raise FileNotFoundError(f"No pkl files found under {input_path}")

    samples = []
    failures = []
    for file_idx, path in enumerate(paths, start=1):
        try:
            qpos, source_fps = _load_gmr_qpos(path, target_fps)
            for chunk_start, chunk in _iter_chunks(qpos, args.chunk_frames, args.overlap_frames):
                if chunk.shape[0] < min_frames:
                    continue
                motion = _qpos_to_motion_feature(converter, motion_rep, chunk, args.device)
                samples.append({
                    "keyid": f"{path.relative_to(input_path if input_path.is_dir() else input_path.parent)}::{chunk_start}",
                    "motion": motion,
                    "source_fps": source_fps,
                })
        except Exception as exc:
            failures.append((str(path), repr(exc)))

        if file_idx % args.log_every == 0:
            print(f"processed={file_idx}/{len(paths)} samples={len(samples)} failures={len(failures)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "samples": samples,
        "source": str(input_path),
        "target_fps": target_fps,
        "motion_dim": len(motion_rep.dual_rep.global_motion_rep.indices["all"]),
        "schema": "gmr_pingpong_g1_pkl",
        "num_failures": len(failures),
        "failures": failures[: args.max_saved_failures],
    }
    torch.save(state, output)
    print(f"saved {len(samples)} samples from {len(paths)} files to {output}")
    if failures:
        print(f"failures: {len(failures)}; first failure: {failures[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/home/zhipy/project/GMR/motion_data/PPP_pingpong_g1_gmr_mirror_right")
    parser.add_argument("--output", default="datasets/gmr_pingpong_g1_motion_features/train_tt.pt")
    parser.add_argument("--hparams", default="out/motionbricks_pose/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", default="assets/skeletons/g1/g1.xml")
    parser.add_argument("--target_fps", type=int, default=None)
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_tokens", type=int, default=4)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--chunk_frames", type=int, default=0, help="0 keeps each pkl as one sample.")
    parser.add_argument("--overlap_frames", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_saved_failures", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
