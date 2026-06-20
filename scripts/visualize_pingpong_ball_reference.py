#!/usr/bin/env python3
"""Visualize ball trajectories saved by generate_pingpong_ball_reference_buffer.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def _draw_table(ax, *, table_z: float = 0.76, ball_radius: float = 0.02) -> None:
    z = table_z + ball_radius
    table = np.array(
        [
            [0.50, -0.76, z],
            [3.24, -0.76, z],
            [3.24, 0.76, z],
            [0.50, 0.76, z],
            [0.50, -0.76, z],
        ],
        dtype=np.float32,
    )
    ax.plot(table[:, 0], table[:, 1], table[:, 2], color="black", linewidth=1.2, label="table edge")
    ax.plot([1.87, 1.87], [-0.76, 0.76], [0.91, 0.91], color="gray", linewidth=2.0, label="net")
    ax.plot([0.3, 0.3], [-0.76, 0.76], [z, z], color="tab:purple", linewidth=1.4, label="hit plane x=0.3")


def _command_index(data: dict[str, np.ndarray], frame_idx: int) -> int:
    starts = data["command_start_frames"].astype(np.int64)
    return int(np.clip(np.searchsorted(starts, frame_idx, side="right") - 1, 0, len(starts) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Generated npz file, ready dir, or output dir.")
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--show_frames", type=int, default=0, help="Limit plotted frames; 0 means all.")
    parser.add_argument("--save", default=None, help="Optional image output path.")
    args = parser.parse_args()

    files = _discover_npz(Path(args.path))
    path = files[int(np.clip(args.clip, 0, len(files) - 1))]
    data = {key: value for key, value in np.load(path).items()}
    if "ball_pos_w" not in data:
        raise KeyError(f"{path} does not contain ball_pos_w. Regenerate with the ball reference script.")

    ball = np.asarray(data["ball_pos_w"], dtype=np.float32)
    if args.show_frames > 0:
        ball = ball[: args.show_frames]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
    starts = data["command_start_frames"].astype(np.int64)
    ends = data["command_end_frames"].astype(np.int64)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_table(ax)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(starts), 1)))
    for cmd_idx, start in enumerate(starts):
        end = int(ends[cmd_idx]) if cmd_idx < len(ends) else ball.shape[0]
        start = int(np.clip(start, 0, ball.shape[0] - 1))
        end = int(np.clip(end, start + 1, ball.shape[0]))
        seg = ball[start:end]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[cmd_idx % len(colors)], linewidth=1.6)

    if "ball_start_pos" in data:
        starts_pos = np.asarray(data["ball_start_pos"], dtype=np.float32)
        ax.scatter(starts_pos[:, 0], starts_pos[:, 1], starts_pos[:, 2], c="tab:blue", s=35, label="ball start")
    if "ball_hit_pos" in data:
        hit_pos = np.asarray(data["ball_hit_pos"], dtype=np.float32)
        ax.scatter(hit_pos[:, 0], hit_pos[:, 1], hit_pos[:, 2], c="tab:red", s=45, label="ball hit plane")
    if "hit_pos" in data:
        target = np.asarray(data["hit_pos"], dtype=np.float32)
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], c="gold", s=22, label="motion target")

    title = path.name
    if "ball_hit_time_s" in data:
        t = np.asarray(data["ball_hit_time_s"], dtype=np.float32)
        title += f" | ball_hit_t mean={float(t.mean()):.3f}s"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(0.0, 3.35)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.65, 1.55)
    ax.view_init(elev=22, azim=-120)
    ax.legend(loc="upper right")
    ax.text2D(0.02, 0.02, f"fps={fps:.0f}, frames={ball.shape[0]}, commands={len(starts)}", transform=ax.transAxes)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
