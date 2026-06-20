#!/usr/bin/env python3
"""Report or move reference-buffer clips with abnormal motion values."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def discover_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.npz"))
    if not files:
        files = sorted(path.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files


def quality_metrics(data: np.lib.npyio.NpzFile) -> dict[str, float]:
    metrics = {
        "max_joint_jump_rad": 0.0,
        "max_joint_vel_abs": 0.0,
        "max_body_lin_speed": 0.0,
        "max_body_ang_speed": 0.0,
        "nonfinite_count": 0.0,
    }
    for key in data.files:
        arr = np.asarray(data[key])
        if np.issubdtype(arr.dtype, np.number):
            metrics["nonfinite_count"] += float((~np.isfinite(arr)).sum())
    if "joint_pos" in data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        if joint_pos.shape[0] > 1:
            metrics["max_joint_jump_rad"] = float(np.max(np.abs(np.diff(joint_pos, axis=0))))
    if "joint_vel" in data:
        metrics["max_joint_vel_abs"] = float(np.max(np.abs(np.asarray(data["joint_vel"], dtype=np.float64))))
    if "body_lin_vel_w" in data:
        metrics["max_body_lin_speed"] = float(
            np.max(np.linalg.norm(np.asarray(data["body_lin_vel_w"], dtype=np.float64), axis=-1)))
    if "body_ang_vel_w" in data:
        metrics["max_body_ang_speed"] = float(
            np.max(np.linalg.norm(np.asarray(data["body_ang_vel_w"], dtype=np.float64), axis=-1)))
    return metrics


def rejection_reasons(metrics: dict[str, float], args) -> list[str]:
    reasons = []
    if metrics["nonfinite_count"] > 0:
        reasons.append(f"nonfinite={int(metrics['nonfinite_count'])}")
    if metrics["max_joint_jump_rad"] > args.max_joint_jump_rad:
        reasons.append(f"joint_jump={metrics['max_joint_jump_rad']:.4f}>{args.max_joint_jump_rad:.4f}")
    if metrics["max_joint_vel_abs"] > args.max_joint_vel_rad_s:
        reasons.append(f"joint_vel={metrics['max_joint_vel_abs']:.3f}>{args.max_joint_vel_rad_s:.3f}")
    if metrics["max_body_lin_speed"] > args.max_body_lin_speed_m_s:
        reasons.append(f"body_lin={metrics['max_body_lin_speed']:.3f}>{args.max_body_lin_speed_m_s:.3f}")
    if metrics["max_body_ang_speed"] > args.max_body_ang_speed_rad_s:
        reasons.append(f"body_ang={metrics['max_body_ang_speed']:.3f}>{args.max_body_ang_speed_rad_s:.3f}")
    return reasons


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(root / "out/reference_5_tokens/pingpong_fk/ready"),
        help="Reference npz file or directory.",
    )
    parser.add_argument("--max_joint_jump_rad", type=float, default=0.3)
    parser.add_argument("--max_joint_vel_rad_s", type=float, default=15.0)
    parser.add_argument("--max_body_lin_speed_m_s", type=float, default=8.0)
    parser.add_argument("--max_body_ang_speed_rad_s", type=float, default=15.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--move_bad", action="store_true", help="Move rejected files instead of only reporting.")
    parser.add_argument(
        "--rejected_dir",
        default=None,
        help="Destination for --move_bad. Defaults to sibling rejected_quality/.",
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    files = discover_npz(path)
    rejected = []
    accepted = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as data:
            metrics = quality_metrics(data)
        reasons = rejection_reasons(metrics, args)
        if reasons:
            rejected.append((file_path, metrics, reasons))
        else:
            accepted += 1

    print(f"scanned={len(files)} accepted={accepted} rejected={len(rejected)}")
    for metric in [
        "max_joint_jump_rad",
        "max_joint_vel_abs",
        "max_body_lin_speed",
        "max_body_ang_speed",
    ]:
        rows = sorted(rejected, key=lambda item: item[1][metric], reverse=True)
        if rows:
            print(f"\ntop rejected by {metric}:")
            for file_path, metrics, reasons in rows[:args.top_k]:
                print(
                    f"{file_path} {metric}={metrics[metric]:.6g} "
                    f"reasons={';'.join(reasons)}"
                )

    if args.move_bad and rejected:
        rejected_dir = (
            Path(args.rejected_dir).expanduser()
            if args.rejected_dir is not None
            else path.parent / "rejected_quality"
        )
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for file_path, _, _ in rejected:
            target = rejected_dir / file_path.name
            if target.exists():
                target = rejected_dir / f"{file_path.stem}_{abs(hash(file_path))}{file_path.suffix}"
            shutil.move(str(file_path), str(target))
        print(f"moved {len(rejected)} files to {rejected_dir}")


if __name__ == "__main__":
    main()
