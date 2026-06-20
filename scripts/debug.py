#!/usr/bin/env python3
"""Print per-file minimum time_to_hit_s values for generated reference motions."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import numpy as np


def scan_file(path: Path) -> tuple[float, int]:
    with np.load(path, allow_pickle=False) as data:
        if "time_to_hit_s" not in data:
            raise KeyError("missing time_to_hit_s")
        time_to_hit = np.asarray(data["time_to_hit_s"], dtype=np.float32)
    if time_to_hit.size == 0:
        raise ValueError("empty time_to_hit_s")
    frame_idx = int(np.nanargmin(time_to_hit))
    return float(time_to_hit[frame_idx]), frame_idx


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "motion_dir",
        nargs="?",
        default=str(root / "out/reference_buffer/pingpong_fk/ready"),
        help="Directory containing generated .npz reference motion files.",
    )
    parser.add_argument("--global-only", action="store_true", help="Only print the global minimum.")
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir).expanduser()
    paths = sorted(motion_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {motion_dir}")

    global_min = float("inf")
    global_path: Path | None = None
    global_frame = -1

    for path in paths:
        try:
            file_min, frame_idx = scan_file(path)
        except Exception as exc:
            print(f"{path.name}: ERROR {exc}")
            continue
        if not args.global_only:
            # print(f"{path.name}: min_time_to_hit_s={file_min:.6f}, frame={frame_idx}")
            print(file_min, end=" ")
        if file_min < global_min:
            global_min = file_min
            global_path = path
            global_frame = frame_idx

    if global_path is not None:
        print(
            global_min, end=" "
            # f"\nGLOBAL_MIN: , "
            # f"file={global_path.name}, frame={global_frame}"
        )


if __name__ == "__main__":
    main()
