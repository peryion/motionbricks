#!/usr/bin/env python3
"""Evaluate the pretrained pose VQ-VAE reconstruction on ping-pong references."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch as t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.helper.pl_util import load_motion_rep
from pingpong_demo_utils import _load_model_from_training_config, _resolve_training_config_for_ckpt
from visualize_pingpong_reference_buffer import ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB


BODY_IDS_FOR_ISAACLAB_G1 = np.arange(1, 31, dtype=np.int32)


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def _discover_npz(path: Path, limit: int) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not files and path.is_dir():
        files = sorted(path.glob("**/*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found under {path}")
    return files[:limit]


def _qpos_from_reference_npz(data: dict[str, np.ndarray]) -> np.ndarray:
    root_pos = np.asarray(data["body_pos_w"], dtype=np.float32)[:, 0, :3]
    root_quat = np.asarray(data["body_quat_w"], dtype=np.float32)[:, 0, :4]
    joint_pos_mujoco = np.asarray(data["joint_pos"], dtype=np.float32)[:, ISAACLAB_TO_MUJOCO]
    return np.concatenate([root_pos, root_quat, joint_pos_mujoco], axis=-1).astype(np.float32)


def _reference_payload_from_qpos(
    qpos: np.ndarray,
    template: dict[str, np.ndarray],
    model: mujoco.MjModel,
    fps: float,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    body_pos = np.zeros((qpos.shape[0], len(BODY_IDS_FOR_ISAACLAB_G1), 3), dtype=np.float32)
    body_quat = np.zeros((qpos.shape[0], len(BODY_IDS_FOR_ISAACLAB_G1), 4), dtype=np.float32)
    for frame_idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos[frame_idx] = data.xpos[BODY_IDS_FOR_ISAACLAB_G1]
        body_quat[frame_idx] = data.xquat[BODY_IDS_FOR_ISAACLAB_G1]
    payload = dict(template)
    payload["fps"] = np.asarray([fps], dtype=np.float32)
    payload["joint_pos"] = qpos[:, 7:36][:, MUJOCO_TO_ISAACLAB].astype(np.float32)
    payload["joint_vel"] = np.zeros_like(payload["joint_pos"], dtype=np.float32)
    if qpos.shape[0] > 1:
        payload["joint_vel"][:-1] = (payload["joint_pos"][1:] - payload["joint_pos"][:-1]) * fps
        payload["joint_vel"][-1] = payload["joint_vel"][-2]
    payload["body_pos_w"] = body_pos
    payload["body_quat_w"] = body_quat
    payload["root_pos_w"] = body_pos[:, 0, :3]
    payload["root_quat_w"] = body_quat[:, 0, :4]
    return payload


def _load_pose_model(cfg) -> t.nn.Module:
    pose_ckpt = _resolve_path(cfg.model.pose_ckpt)
    pose_config = _resolve_training_config_for_ckpt(pose_ckpt, _resolve_path(cfg.model.pose_config))
    if pose_ckpt is None or pose_config is None:
        raise ValueError("configs/pingpong_g1.yaml must provide model.pose_ckpt and model.pose_config.")
    old_load = t.load

    def load_on_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return old_load(*args, **kwargs)

    t.load = load_on_cpu
    try:
        return _load_model_from_training_config(pose_config, pose_ckpt, "pose").eval()
    finally:
        t.load = old_load


def _load_direct_vqvae_pose_net(config_path: str, ckpt_path: str) -> t.nn.Module:
    conf = OmegaConf.load(config_path)
    if "train" in conf and "model" in conf:
        conf = conf.model

    motion_rep = load_motion_rep(conf)
    model_conf = conf.model
    pose_vqvae_motion_rep = getattr(model_conf, "pose_vqvae_motion_rep", "local")
    pose_motion_rep = (
        motion_rep.dual_rep.local_motion_rep
        if pose_vqvae_motion_rep == "local"
        else motion_rep.dual_rep.global_motion_rep
    )
    pose_net = instantiate(model_conf.pose_vqvae_network, motion_rep=pose_motion_rep)

    state = t.load(ckpt_path, map_location="cpu")["state_dict"]
    pose_state = {
        key.removeprefix("pose_net."): value
        for key, value in state.items()
        if key.startswith("pose_net.")
    }
    target_state = pose_net.state_dict()
    matched = {
        key: value
        for key, value in pose_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    pose_net.load_state_dict(matched, strict=False)
    print(f"Loaded direct VQ-VAE pose_net: {ckpt_path}")
    print(f"  matched tensors: {len(matched)} / {len(target_state)}")
    return pose_net.eval()


def _load_pose_net_from_cfg_or_vqvae(cfg, vqvae_config: str | None = None, vqvae_ckpt: str | None = None) -> t.nn.Module:
    if vqvae_ckpt:
        if not vqvae_config:
            vqvae_config = str(Path(vqvae_ckpt).resolve().parent.parent / "config.yaml")
        return _load_direct_vqvae_pose_net(_resolve_path(vqvae_config), _resolve_path(vqvae_ckpt))
    return _load_pose_model(cfg).supporting_nets["pose_net"].eval()


def _qpos_to_motion_features(qpos: np.ndarray, converter, motion_rep, device: str) -> t.Tensor:
    qpos_t = t.from_numpy(qpos).to(device).view(1, qpos.shape[0], qpos.shape[1])
    joint_pos, joint_rot = converter.convert_mujoco_qpos_to_motion_transforms(qpos_t)
    features = motion_rep(
        {"posed_joints": joint_pos, "global_joint_rots": joint_rot},
        to_normalize=True,
        lengths=t.tensor([qpos.shape[0]], device=device),
    )
    return features


def _reconstruct_with_pose_vqvae(features: t.Tensor, pose_net) -> t.Tensor:
    frames_per_token = int(2 ** getattr(pose_net, "_down_t", 2))
    usable_frames = (features.shape[1] // frames_per_token) * frames_per_token
    if usable_frames <= 0:
        raise ValueError(f"Motion is too short for VQ-VAE reconstruction: {features.shape[1]} frames")
    if usable_frames != features.shape[1]:
        print(
            f"Cropping VQ-VAE input from {features.shape[1]} to {usable_frames} frames "
            f"to align with {frames_per_token} frames/token."
        )
        features = features[:, :usable_frames]

    pose_features = (
        pose_net.motion_rep.get_feature_subset(features, pose_net.motion_rep.name)
        if hasattr(pose_net.motion_rep, "get_feature_subset")
        else features
    )
    external_cond = extract_feature_from_motion_rep(
        pose_features,
        pose_net.motion_rep,
        pose_net.decoder_external_cond_feature_mode,
    )
    with t.no_grad():
        try:
            out = pose_net(
                pose_features,
                target_cond=None,
                has_target_cond=None,
                external_cond=external_cond,
            )
        except AssertionError as exc:
            raise AssertionError(
                "VQ-VAE decoder condition length mismatch: "
                f"pose_features={tuple(pose_features.shape)}, "
                f"external_cond={tuple(external_cond.shape)}, "
                f"frames_per_token={frames_per_token}"
            ) from exc
    return out["recon_state"]


def _racket_metrics(model: mujoco.MjModel, qpos: np.ndarray, recon_qpos: np.ndarray, site_name: str) -> dict[str, float]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return {}
    data = mujoco.MjData(model)
    recon_data = mujoco.MjData(model)
    pos_err = []
    for i in range(min(qpos.shape[0], recon_qpos.shape[0])):
        data.qpos[:] = qpos[i]
        recon_data.qpos[:] = recon_qpos[i]
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(model, recon_data)
        pos_err.append(np.linalg.norm(data.site_xpos[site_id] - recon_data.site_xpos[site_id]))
    pos_err = np.asarray(pos_err)
    return {
        "racket_pos_mean_m": float(pos_err.mean()),
        "racket_pos_p95_m": float(np.percentile(pos_err, 95)),
        "racket_pos_max_m": float(pos_err.max()),
    }


def _metrics(model: mujoco.MjModel, qpos: np.ndarray, recon_qpos: np.ndarray, site_name: str) -> dict[str, float]:
    num_frames = min(qpos.shape[0], recon_qpos.shape[0])
    qpos = qpos[:num_frames]
    recon_qpos = recon_qpos[:num_frames]
    joint_abs = np.abs(qpos[:, 7:36] - recon_qpos[:, 7:36])
    root_xy = np.linalg.norm(qpos[:, :2] - recon_qpos[:, :2], axis=-1)
    root_z = np.abs(qpos[:, 2] - recon_qpos[:, 2])
    out = {
        "joint_mae_rad": float(joint_abs.mean()),
        "joint_p95_rad": float(np.percentile(joint_abs, 95)),
        "joint_max_rad": float(joint_abs.max()),
        "root_xy_mean_m": float(root_xy.mean()),
        "root_xy_p95_m": float(np.percentile(root_xy, 95)),
        "root_z_mean_m": float(root_z.mean()),
    }
    out.update(_racket_metrics(model, qpos, recon_qpos, site_name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pingpong_g1.yaml"))
    parser.add_argument("--path", default=str(ROOT / "out/reference_buffer/pingpong_fk/ready"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--site_name", default="paddle_center")
    parser.add_argument("--save_npz", default=None, help="Optional path to save the first reconstructed clip.")
    parser.add_argument("--vqvae_ckpt", default=None, help="Direct fine-tuned VQ-VAE Lightning checkpoint.")
    parser.add_argument("--vqvae_config", default=None, help="Config saved next to the fine-tuned VQ-VAE checkpoint.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pose_net = _load_pose_net_from_cfg_or_vqvae(cfg, args.vqvae_config, args.vqvae_ckpt).to(args.device)
    motion_rep = pose_net.motion_rep.to(args.device)
    converter = get_mujoco_converter(motion_rep, _resolve_path(cfg.assets.skeleton_xml)).to(args.device)
    model = mujoco.MjModel.from_xml_path(_resolve_path(cfg.assets.humanoid_xml))

    files = _discover_npz(Path(args.path), args.limit)
    all_metrics = []
    first_recon_payload = None
    for path in files:
        loaded = np.load(path, allow_pickle=True)
        data = {key: loaded[key] for key in loaded.files}
        qpos = _qpos_from_reference_npz(data)
        features = _qpos_to_motion_features(qpos, converter, motion_rep, args.device)
        recon_pose = _reconstruct_with_pose_vqvae(features, pose_net)
        recon_qpos = converter.convert_motion_features_to_mujoco_qpos(
            recon_pose, motion_rep, is_normalized=True, root_quat_w_first=False)
        recon_qpos = recon_qpos.detach().cpu().numpy()[0].astype(np.float32)
        root_rot = recon_qpos[:, 3:7].copy()
        recon_qpos[:, 3:7] = root_rot[:, [3, 0, 1, 2]]
        m = _metrics(model, qpos, recon_qpos, args.site_name)
        all_metrics.append(m)
        print(
            f"{path.name}: joint_mae={m['joint_mae_rad']:.4f} rad, "
            f"joint_p95={m['joint_p95_rad']:.4f}, root_xy={m['root_xy_mean_m']:.3f} m"
            + ("" if "racket_pos_mean_m" not in m else f", racket={m['racket_pos_mean_m']:.3f} m")
        )
        if first_recon_payload is None:
            fps = float(np.asarray(data.get("fps", [50.0])).reshape(-1)[0])
            first_recon_payload = _reference_payload_from_qpos(recon_qpos, data, model, fps)

    keys = sorted(all_metrics[0].keys())
    print("\nAverages:")
    for key in keys:
        vals = np.asarray([m[key] for m in all_metrics], dtype=np.float64)
        print(f"  {key}: mean={vals.mean():.6f}, max={vals.max():.6f}")

    if args.save_npz:
        out = Path(args.save_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **first_recon_payload)
        print(f"\nSaved reconstructed first clip: {out}")


if __name__ == "__main__":
    main()
