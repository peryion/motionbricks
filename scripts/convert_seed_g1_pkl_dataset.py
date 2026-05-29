#!/usr/bin/env python3
"""Convert zlib-compressed SEED G1 pkl motions to MotionBricks training features."""

from __future__ import annotations

import argparse
import io
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep


def _import_joblib():
    try:
        import joblib
        return joblib
    except ModuleNotFoundError:
        candidates = [
            Path.home() / "miniconda3/lib/python3.13/site-packages",
            Path.home() / "miniconda3/lib/python3.12/site-packages",
            Path.home() / "miniconda3/lib/python3.11/site-packages",
        ]
        for candidate in candidates:
            if (candidate / "joblib").exists():
                sys.path.append(str(candidate))
                import joblib
                return joblib
        raise


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_seed_pkl(path: Path) -> dict:
    joblib = _import_joblib()
    raw = path.read_bytes()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    obj = joblib.load(io.BytesIO(raw))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not contain a dict")
    return obj


def _resample_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-5:
        return np.arange(num_frames, dtype=np.int64)
    target_frames = max(1, int(np.floor(num_frames * target_fps / source_fps)))
    indices = np.round(np.arange(target_frames) * source_fps / target_fps).astype(np.int64)
    return np.clip(indices, 0, num_frames - 1)


def _entry_to_qpos(entry: dict, target_fps: int) -> np.ndarray:
    root_pos = np.asarray(entry["root_trans_offset"], dtype=np.float32)
    root_quat_xyzw = np.asarray(entry["root_rot"], dtype=np.float32)
    dof = np.asarray(entry["dof"], dtype=np.float32)
    source_fps = float(entry.get("fps", target_fps))

    if root_pos.shape[0] != root_quat_xyzw.shape[0] or root_pos.shape[0] != dof.shape[0]:
        raise ValueError("root_trans_offset, root_rot, and dof have inconsistent lengths")
    if root_quat_xyzw.shape[-1] != 4 or dof.shape[-1] != 29:
        raise ValueError(f"Expected root_rot (*,4) and dof (*,29), got {root_quat_xyzw.shape}, {dof.shape}")

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof], axis=-1).astype(np.float32)
    return qpos[_resample_indices(qpos.shape[0], source_fps, target_fps)]


def _qpos_to_motion_feature(converter, motion_rep, qpos: np.ndarray, device: str) -> torch.Tensor:
    with torch.no_grad():
        qpos_tensor = torch.from_numpy(qpos).float()[None].to(device)
        joint_pos, joint_rot = converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
        lengths = torch.tensor([qpos.shape[0]], device=device)
        features = motion_rep.dual_rep.global_motion_rep(
            {
                "posed_joints": joint_pos,
                "global_joint_rots": joint_rot,
            },
            to_normalize=True,
            lengths=lengths,
        )
    return features[0].detach().cpu()


def convert(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = _resolve(root, args.input_dir)
    output = _resolve(root, args.output)
    conf = OmegaConf.load(_resolve(root, args.hparams))
    motion_rep = load_motion_rep(conf).to(args.device)
    converter = get_mujoco_converter(motion_rep, str(_resolve(root, args.skeleton_xml))).to(args.device)
    target_fps = int(args.target_fps or motion_rep.fps)
    min_frames = int(args.min_tokens * args.frames_per_token + 1)

    paths = sorted(input_dir.rglob(args.pattern))
    if args.max_files is not None:
        paths = paths[:args.max_files]

    samples = []
    failures = []
    for file_idx, path in enumerate(paths, start=1):
        try:
            motions = _load_seed_pkl(path)
            for name, entry in motions.items():
                qpos = _entry_to_qpos(entry, target_fps)
                if qpos.shape[0] < min_frames:
                    continue
                motion = _qpos_to_motion_feature(converter, motion_rep, qpos, args.device)
                samples.append({
                    "keyid": f"{path.relative_to(input_dir)}::{name}",
                    "motion": motion,
                })
        except Exception as exc:
            failures.append((str(path), repr(exc)))

        if file_idx % args.log_every == 0:
            print(f"processed={file_idx}/{len(paths)} samples={len(samples)} failures={len(failures)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "samples": samples,
        "source_dir": str(input_dir),
        "target_fps": target_fps,
        "motion_dim": len(motion_rep.dual_rep.global_motion_rep.indices["all"]),
        "num_failures": len(failures),
        "failures": failures[: args.max_saved_failures],
    }
    torch.save(state, output)
    print(f"saved {len(samples)} samples to {output}")
    if failures:
        print(f"failures: {len(failures)}; first failure: {failures[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="/home/zhipy/Documents/dataset/seed/g1/motion_lib_bones_seed/robot_filtered")
    parser.add_argument("--output", default="datasets/seed_g1_motion_features/train.pt")
    parser.add_argument("--hparams", default="out/motionbricks_pose/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", default="assets/skeletons/g1/g1.xml")
    parser.add_argument("--target_fps", type=int, default=None)
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_tokens", type=int, default=4)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--max_saved_failures", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
