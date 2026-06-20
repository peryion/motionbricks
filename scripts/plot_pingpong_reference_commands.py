#!/usr/bin/env python3
"""Plot hit target positions and velocities from generated ping-pong episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from omegaconf import OmegaConf

from pingpong_command import PingPongCommandState


def _discover_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.npz"))
    if not files:
        files = sorted(path.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files


def _load_commands(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit_pos, hit_vel, stroke_sign = [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            hit_pos.append(np.asarray(data["hit_pos"], dtype=np.float32))
            hit_vel.append(np.asarray(data["hit_vel"], dtype=np.float32))
            stroke_sign.append(np.asarray(data["stroke_sign"], dtype=np.int8))
    return np.concatenate(hit_pos), np.concatenate(hit_vel), np.concatenate(stroke_sign)


def _print_stats(hit_pos: np.ndarray, hit_vel: np.ndarray, stroke_sign: np.ndarray) -> None:
    speed = np.linalg.norm(hit_vel, axis=-1)
    print(f"commands: {hit_pos.shape[0]}")
    print(f"forehand: {(stroke_sign > 0).sum()}, backhand: {(stroke_sign < 0).sum()}")
    print(f"hit x range: {hit_pos[:, 0].min():.3f} .. {hit_pos[:, 0].max():.3f}")
    print(f"hit y range: {hit_pos[:, 1].min():.3f} .. {hit_pos[:, 1].max():.3f}")
    print(f"hit z range: {hit_pos[:, 2].min():.3f} .. {hit_pos[:, 2].max():.3f}")
    print(f"vel x range: {hit_vel[:, 0].min():.3f} .. {hit_vel[:, 0].max():.3f}")
    print(f"vel y range: {hit_vel[:, 1].min():.3f} .. {hit_vel[:, 1].max():.3f}")
    print(f"vel z range: {hit_vel[:, 2].min():.3f} .. {hit_vel[:, 2].max():.3f}")
    print(f"speed range: {speed.min():.3f} .. {speed.max():.3f}")


def _sample_raw_commands(config_path: Path, num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path)["command"], resolve=True)
    cfg = {**cfg, "mixed_use_base_relative_y_split": True}
    state = PingPongCommandState(cfg)
    hit_pos = np.zeros((num_samples, 3), dtype=np.float32)
    hit_vel = np.zeros((num_samples, 3), dtype=np.float32)
    stroke_sign = np.zeros((num_samples,), dtype=np.int8)
    for idx in range(num_samples):
        state.resample(base_y=0.0)
        hit_pos[idx] = state.target_hit_pos
        hit_vel[idx] = state.target_hit_vel
        stroke_sign[idx] = 1 if state.is_forehand else -1
    return hit_pos, hit_vel, stroke_sign


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(root / "out/reference_5_tokens/pingpong_fk/ready"),
        help="Reference npz file or directory.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "out/reference_5_tokens/pingpong_fk/command_distribution.png"),
        help="Output image path.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument("--max_arrows", type=int, default=600, help="Maximum velocity arrows to draw.")
    parser.add_argument("--arrow_scale", type=float, default=0.18, help="Velocity arrow scale in y-z plot.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/pingpong_g1.yaml"),
        help="Config used to sample raw commands for comparison.",
    )
    parser.add_argument("--raw_samples", type=int, default=20000, help="Number of raw commands to overlay.")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = _discover_npz(Path(args.path).expanduser())
    hit_pos, hit_vel, stroke_sign = _load_commands(files)
    raw_hit_pos, raw_hit_vel, raw_stroke_sign = _sample_raw_commands(Path(args.config), args.raw_samples)
    _print_stats(hit_pos, hit_vel, stroke_sign)

    fore = stroke_sign > 0
    back = stroke_sign < 0
    speed = np.linalg.norm(hit_vel, axis=-1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(raw_hit_pos[:, 1], raw_hit_pos[:, 2], s=5, c="#bdbdbd", alpha=0.12, label="raw sampler")
    ax.scatter(hit_pos[fore, 1], hit_pos[fore, 2], s=14, c="#1f77b4", alpha=0.55, label="forehand")
    ax.scatter(hit_pos[back, 1], hit_pos[back, 2], s=14, c="#d62728", alpha=0.55, label="backhand")
    if hit_pos.shape[0] > 0 and args.max_arrows > 0:
        step = max(1, int(np.ceil(hit_pos.shape[0] / args.max_arrows)))
        idx = np.arange(0, hit_pos.shape[0], step)
        ax.quiver(
            hit_pos[idx, 1],
            hit_pos[idx, 2],
            hit_vel[idx, 1] * args.arrow_scale,
            hit_vel[idx, 2] * args.arrow_scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.002,
            alpha=0.35,
            color="#333333",
        )
    ax.set_xlabel("hit y (m)")
    ax.set_ylabel("hit z (m)")
    ax.set_title("Hit target y-z distribution with velocity y-z arrows")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(hit_pos[:, 1], hit_vel[:, 1], s=12, c=speed, cmap="viridis", alpha=0.6)
    ax.set_xlabel("hit y (m)")
    ax.set_ylabel("hit vel y (m/s)")
    ax.set_title("Lateral hit position vs lateral velocity")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 2]
    y_bins = np.linspace(
        min(raw_hit_pos[:, 1].min(), hit_pos[:, 1].min()),
        max(raw_hit_pos[:, 1].max(), hit_pos[:, 1].max()),
        41,
    )
    z_bins = np.linspace(
        min(raw_hit_pos[:, 2].min(), hit_pos[:, 2].min()),
        max(raw_hit_pos[:, 2].max(), hit_pos[:, 2].max()),
        41,
    )
    raw_hist, _, _ = np.histogram2d(raw_hit_pos[:, 1], raw_hit_pos[:, 2], bins=(y_bins, z_bins))
    acc_hist, _, _ = np.histogram2d(hit_pos[:, 1], hit_pos[:, 2], bins=(y_bins, z_bins))
    ratio = np.divide(acc_hist, raw_hist, out=np.full_like(acc_hist, np.nan), where=raw_hist > 0)
    image = ax.imshow(
        ratio.T,
        origin="lower",
        extent=[y_bins[0], y_bins[-1], z_bins[0], z_bins[-1]],
        aspect="auto",
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, label="accepted/raw bin ratio")
    ax.set_xlabel("hit y (m)")
    ax.set_ylabel("hit z (m)")
    ax.set_title("Accepted density relative to raw sampler")

    ax = axes[1, 0]
    ax.hist(raw_hit_pos[:, 1], bins=40, alpha=0.25, color="#7f7f7f", label="raw y")
    ax.hist(hit_pos[fore, 1], bins=40, alpha=0.55, color="#1f77b4", label="forehand y")
    ax.hist(hit_pos[back, 1], bins=40, alpha=0.55, color="#d62728", label="backhand y")
    ax.set_xlabel("hit y (m)")
    ax.set_ylabel("count")
    ax.set_title("Hit y histogram")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.hist(raw_hit_pos[:, 2], bins=40, alpha=0.25, color="#7f7f7f", label="raw hit z")
    ax.hist(hit_pos[:, 2], bins=40, alpha=0.55, color="#2ca02c", label="hit z")
    ax2 = ax.twinx()
    ax2.hist(speed, bins=40, alpha=0.35, color="#9467bd", label="speed")
    ax.set_xlabel("value")
    ax.set_ylabel("hit z count")
    ax2.set_ylabel("speed count")
    ax.set_title("Hit z and speed histograms")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    raw_speed = np.linalg.norm(raw_hit_vel, axis=-1)
    ax.hist(raw_speed, bins=40, alpha=0.3, color="#7f7f7f", label="raw speed")
    ax.hist(speed, bins=40, alpha=0.55, color="#9467bd", label="accepted speed")
    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("count")
    ax.set_title("Raw vs accepted speed")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"saved: {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
