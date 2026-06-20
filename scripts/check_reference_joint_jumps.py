#!/usr/bin/env python3
"""Find large frame-to-frame joint position jumps in reference buffer episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _discover_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.npz"))
    if not files:
        files = sorted(path.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files


def _nearest_boundary(frame: int, data: np.lib.npyio.NpzFile) -> tuple[int, int]:
    boundaries = []
    if "command_start_frames" in data:
        boundaries.extend(np.asarray(data["command_start_frames"], dtype=np.int64).tolist())
    if "command_end_frames" in data:
        boundaries.extend(np.asarray(data["command_end_frames"], dtype=np.int64).tolist())
    if not boundaries:
        return -1, -1
    arr = np.asarray(boundaries, dtype=np.int64)
    idx = int(np.argmin(np.abs(arr - frame)))
    return int(arr[idx]), int(abs(arr[idx] - frame))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(root / "out/reference_5_tokens/pingpong_fk/ready"),
        help="Reference npz file or directory.",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Jump threshold in radians.")
    parser.add_argument("--top_k", type=int, default=30, help="Number of largest jumps to print.")
    parser.add_argument("--boundary_window", type=int, default=10, help="Frames counted as near a command boundary.")
    args = parser.parse_args()

    files = _discover_npz(Path(args.path).expanduser())
    rows = []
    per_file_max = []
    over_threshold_joint_indices = []
    near_boundary_count = 0
    over_threshold_count = 0
    joint_names = None

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "joint_pos" not in data:
                continue
            if joint_names is None and "joint_names" in data:
                joint_names = [str(name) for name in data["joint_names"]]

            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            if joint_pos.shape[0] < 2:
                continue

            jumps = np.abs(np.diff(joint_pos, axis=0))
            flat_idx = int(np.argmax(jumps))
            frame0, joint_idx = np.unravel_index(flat_idx, jumps.shape)
            max_jump = float(jumps[frame0, joint_idx])
            per_file_max.append(max_jump)

            boundary, boundary_dist = _nearest_boundary(frame0 + 1, data)
            time_to_hit = (
                float(data["time_to_hit_s"][frame0 + 1])
                if "time_to_hit_s" in data and frame0 + 1 < len(data["time_to_hit_s"])
                else float("nan")
            )
            rows.append((
                max_jump,
                path,
                int(frame0),
                int(frame0 + 1),
                int(joint_idx),
                float(joint_pos[frame0, joint_idx]),
                float(joint_pos[frame0 + 1, joint_idx]),
                boundary,
                boundary_dist,
                time_to_hit,
            ))

            above = np.argwhere(jumps > args.threshold)
            over_threshold_count += int(above.shape[0])
            for frame_idx, joint_idx in above:
                over_threshold_joint_indices.append(int(joint_idx))
                _, dist = _nearest_boundary(int(frame_idx) + 1, data)
                if dist >= 0 and dist <= args.boundary_window:
                    near_boundary_count += 1

    rows.sort(reverse=True, key=lambda item: item[0])
    per_file_max_arr = np.asarray(per_file_max, dtype=np.float64)

    print(f"files: {len(files)}, scanned: {len(per_file_max)}")
    if per_file_max_arr.size:
        print(
            "per-file max jump rad: "
            f"min={per_file_max_arr.min():.4f}, "
            f"p50={np.percentile(per_file_max_arr, 50):.4f}, "
            f"p90={np.percentile(per_file_max_arr, 90):.4f}, "
            f"p99={np.percentile(per_file_max_arr, 99):.4f}, "
            f"max={per_file_max_arr.max():.4f}"
        )

    print(f"jumps > {args.threshold:.3f} rad: {over_threshold_count}")
    if over_threshold_count:
        print(
            f"near command boundary <= {args.boundary_window} frames: "
            f"{near_boundary_count}/{over_threshold_count}"
        )
        unique, counts = np.unique(np.asarray(over_threshold_joint_indices), return_counts=True)
        print("joint counts:")
        for joint_idx, count in sorted(zip(unique, counts), key=lambda item: -item[1])[:20]:
            name = joint_names[int(joint_idx)] if joint_names and int(joint_idx) < len(joint_names) else "unknown"
            print(f"  joint_idx={int(joint_idx):02d} {name}: {int(count)}")

    print(f"top {min(args.top_k, len(rows))} per-file max jumps:")
    for item in rows[:args.top_k]:
        jump, path, frame0, frame1, joint_idx, v0, v1, boundary, boundary_dist, time_to_hit = item
        name = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else "unknown"
        print(
            f"jump={jump:.4f} rad file={path.name} frame={frame0}->{frame1} "
            f"joint_idx={joint_idx}:{name} value={v0:.4f}->{v1:.4f} "
            f"nearest_boundary={boundary} boundary_dist={boundary_dist} "
            f"time_to_hit={time_to_hit:.3f}s"
        )


if __name__ == "__main__":
    main()
