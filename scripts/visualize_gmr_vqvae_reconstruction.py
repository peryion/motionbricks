#!/usr/bin/env python3
"""Visualize original GMR pkl motion and its pose VQ-VAE reconstruction."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf
import torch as t

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_pingpong_vqvae_reconstruction import (  # noqa: E402
    _load_pose_net_from_cfg_or_vqvae,
    _qpos_to_motion_features,
    _reconstruct_with_pose_vqvae,
    _resolve_path,
)
from motionbricks.helper.mujoco_helper import get_mujoco_converter  # noqa: E402
from pingpong_demo_utils import disable_mujoco_keyboard_shortcuts  # noqa: E402


def _load_gmr_qpos(path: Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    with path.open("rb") as f:
        data = pickle.load(f)
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    root_quat_wxyz /= np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True).clip(min=1e-8)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    if max_frames is not None and max_frames > 0:
        qpos = qpos[:max_frames]
    return qpos.astype(np.float32), float(data.get("fps", 120.0))


def _reconstruct_qpos(
    qpos: np.ndarray,
    cfg,
    device: str,
    vqvae_config: str | None = None,
    vqvae_ckpt: str | None = None,
) -> np.ndarray:
    pose_net = _load_pose_net_from_cfg_or_vqvae(cfg, vqvae_config, vqvae_ckpt).to(device)
    motion_rep = pose_net.motion_rep.to(device)
    converter = get_mujoco_converter(motion_rep, _resolve_path(cfg.assets.skeleton_xml)).to(device)

    features = _qpos_to_motion_features(qpos, converter, motion_rep, device)
    recon_pose = _reconstruct_with_pose_vqvae(features, pose_net)
    recon_qpos = converter.convert_motion_features_to_mujoco_qpos(
        recon_pose, motion_rep, is_normalized=True, root_quat_w_first=False)
    recon_qpos = recon_qpos.detach().cpu().numpy()[0].astype(np.float32)
    root_rot = recon_qpos[:, 3:7].copy()
    recon_qpos[:, 3:7] = root_rot[:, [3, 0, 1, 2]]
    return recon_qpos


def _draw_recon_ghost(viewer, model, data, opt, pert, qpos: np.ndarray, offset_y: float) -> None:
    q = qpos.copy()
    q[1] += offset_y
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    start = viewer.user_scn.ngeom
    mujoco.mjv_addGeoms(model, data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, viewer.user_scn)
    for geom_idx in range(start, viewer.user_scn.ngeom):
        viewer.user_scn.geoms[geom_idx].rgba[:] = np.array([0.1, 0.45, 1.0, 0.38], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        default="/home/zhipy/project/GMR/motion_data/PPP_pingpong_g1_gmr_mirror_right/Transition_looping_FB.pkl",
    )
    parser.add_argument("--config", default=str(ROOT / "configs/pingpong_g1.yaml"))
    parser.add_argument("--humanoid_xml", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_frames", type=int, default=1200)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--offset_y", type=float, default=1.2)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--vqvae_ckpt", default=None, help="Direct fine-tuned VQ-VAE Lightning checkpoint.")
    parser.add_argument("--vqvae_config", default=None, help="Config saved next to the fine-tuned VQ-VAE checkpoint.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    humanoid_xml = args.humanoid_xml or _resolve_path(cfg.assets.humanoid_xml)
    qpos, fps = _load_gmr_qpos(Path(args.pkl), args.max_frames)
    recon_qpos = _reconstruct_qpos(qpos, cfg, args.device, args.vqvae_config, args.vqvae_ckpt)
    num_frames = min(qpos.shape[0], recon_qpos.shape[0])
    qpos = qpos[:num_frames]
    recon_qpos = recon_qpos[:num_frames]

    joint_err = np.abs(qpos[:, 7:36] - recon_qpos[:, 7:36])
    root_err = np.linalg.norm(qpos[:, :3] - recon_qpos[:, :3], axis=-1)
    print(
        f"frames={qpos.shape[0]} fps={fps:g} "
        f"joint_mae={joint_err.mean():.4f} rad joint_p95={np.percentile(joint_err,95):.4f} "
        f"root_mean={root_err.mean():.4f} m"
    )
    print("viewer: original at real position, reconstruction shifted +Y in blue")
    if args.no_viewer:
        return

    model = mujoco.MjModel.from_xml_path(humanoid_xml)
    model.opt.timestep = 1.0 / fps
    data = mujoco.MjData(model)
    ghost_data = mujoco.MjData(model)
    ghost_opt = mujoco.MjvOption()
    ghost_pert = mujoco.MjvPerturb()
    frame = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        disable_mujoco_keyboard_shortcuts("")
        while viewer.is_running():
            start = time.time()
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            viewer.user_scn.ngeom = 0
            _draw_recon_ghost(viewer, model, ghost_data, ghost_opt, ghost_pert, recon_qpos[frame], args.offset_y)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            frame = (frame + 1) % qpos.shape[0]
            sleep = (1.0 / fps) / max(args.speed, 1e-6) - (time.time() - start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
