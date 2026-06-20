#!/usr/bin/env python3
"""Generate MotionBricks ping-pong reference clips for RL training.

The script runs the ping-pong command controller without a MuJoCo viewer, asks
MotionBricks to synthesize full-body clips, and exports IsaacLab-readable npz
files. It can be used once as an offline dataset builder or kept alive as an
asynchronous producer while RL training consumes the `ready/` directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch as t
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R, Slerp

from pingpong_motion_agent import (
    generate_hit_segment_from_context,
    generate_recovery_segment_from_context,
)
from scripts.pingpong_demo_g1 import (
    Demo,
    parse_args as parse_demo_args,
)


BODY_IDS_FOR_ISAACLAB_G1 = np.arange(1, 31, dtype=np.int32)
MUJOCO_TO_ISAACLAB = np.array(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int64,
)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, t.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    bad = norm.squeeze(-1) < 1e-6
    norm[bad] = 1.0
    quat /= norm
    quat[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] > 1:
        vel[:-1] = (values[1:] - values[:-1]) * fps
        vel[-1] = vel[-2]
    return vel


def _angular_velocity_wxyz(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    ang = np.zeros((*quat_wxyz.shape[:2], 3), dtype=np.float32)
    if quat_wxyz.shape[0] <= 1:
        return ang
    for body_idx in range(quat_wxyz.shape[1]):
        rot = R.from_quat(quat_xyzw[:, body_idx])
        delta = rot[1:] * rot[:-1].inv()
        ang[:-1, body_idx] = delta.as_rotvec().astype(np.float32) * fps
        ang[-1, body_idx] = ang[-2, body_idx]
    return ang


def _interpolate_resample_qpos(qpos: np.ndarray, source_fps: int, target_fps: int) -> np.ndarray:
    """Resample qpos onto the target control-rate time grid with interpolation."""
    if int(source_fps) == int(target_fps):
        return qpos.astype(np.float32, copy=False)
    num_target_frames = max(1, int(round(qpos.shape[0] * float(target_fps) / float(source_fps))))
    source_times = np.arange(qpos.shape[0], dtype=np.float64) / float(source_fps)
    target_times = np.arange(num_target_frames, dtype=np.float64) / float(target_fps)
    target_times = np.clip(target_times, source_times[0], source_times[-1])

    out = np.empty((num_target_frames, qpos.shape[1]), dtype=np.float32)
    for dim in range(qpos.shape[1]):
        if 3 <= dim < 7:
            continue
        out[:, dim] = np.interp(target_times, source_times, qpos[:, dim]).astype(np.float32)

    quat_xyzw = _normalize_quat_wxyz(qpos[:, 3:7])[..., [1, 2, 3, 0]]
    slerp = Slerp(source_times, R.from_quat(quat_xyzw))
    out[:, 3:7] = slerp(target_times).as_quat()[..., [3, 0, 1, 2]].astype(np.float32)
    out[:, 3:7] = _normalize_quat_wxyz(out[:, 3:7])
    return out


def _rescale_frame_index(frame_idx: int, source_fps: int, target_fps: int, num_target_frames: int) -> int:
    scaled = int(round(float(frame_idx) * float(target_fps) / float(source_fps)))
    return int(np.clip(scaled, 0, num_target_frames))


def _rescale_command_records(
    command_records: list[dict[str, np.ndarray]],
    source_fps: int,
    target_fps: int,
    num_target_frames: int,
) -> list[dict[str, np.ndarray]]:
    if int(source_fps) == int(target_fps):
        return command_records
    rescaled = []
    for record in command_records:
        copied = {key: np.array(value, copy=True) for key, value in record.items()}
        copied["command_start_frame"] = np.array(
            [_rescale_frame_index(int(record["command_start_frame"][0]), source_fps, target_fps, num_target_frames)],
            dtype=np.int64,
        )
        copied["command_end_frame"] = np.array(
            [_rescale_frame_index(int(record["command_end_frame"][0]), source_fps, target_fps, num_target_frames)],
            dtype=np.int64,
        )
        rescaled.append(copied)
    return rescaled


def qpos_to_isaaclab_motion_npz(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: int,
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert MotionBricks MuJoCo qpos to the npz schema used by MotionLoader."""
    qpos = np.asarray(qpos, dtype=np.float32)
    mujoco_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    joint_names = np.asarray([mujoco_joint_names[int(idx)] for idx in MUJOCO_TO_ISAACLAB], dtype="U64")
    body_names = np.asarray(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) for body_id in BODY_IDS_FOR_ISAACLAB_G1],
        dtype="U64",
    )
    joint_pos = qpos[:, 7:36][:, MUJOCO_TO_ISAACLAB].astype(np.float32, copy=False)
    joint_vel = _finite_difference(joint_pos, float(fps))

    data = mujoco.MjData(model)
    body_pos = np.zeros((qpos.shape[0], len(BODY_IDS_FOR_ISAACLAB_G1), 3), dtype=np.float32)
    body_quat = np.zeros((qpos.shape[0], len(BODY_IDS_FOR_ISAACLAB_G1), 4), dtype=np.float32)
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        body_pos[frame_idx] = data.xpos[BODY_IDS_FOR_ISAACLAB_G1]
        body_quat[frame_idx] = data.xquat[BODY_IDS_FOR_ISAACLAB_G1]
    body_quat = _normalize_quat_wxyz(body_quat)

    out = {
        "fps": np.array([int(fps)], dtype=np.int64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "joint_names": joint_names,
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _finite_difference(body_pos, float(fps)),
        "body_ang_vel_w": _angular_velocity_wxyz(body_quat, float(fps)),
        "body_names": body_names,
    }
    out.update(metadata)
    return out


def _tail_context_qpos(qpos: np.ndarray, num_frames: int = 4) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[0] >= num_frames:
        return qpos[-num_frames:].copy()
    pad = np.repeat(qpos[:1], num_frames - qpos.shape[0], axis=0)
    return np.concatenate([pad, qpos], axis=0).astype(np.float32, copy=False)


def _generate_no_command_gap(
    demo: Demo,
    num_frames: int,
    controller_dt: float,
    context_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate one recovery segment, then hold its last frame until the next command."""
    if num_frames <= 0:
        return np.zeros((0, demo.mj_model.nq), dtype=np.float32), context_qpos, 0

    recovery = generate_recovery_segment_from_context(demo, context_qpos, controller_dt)

    if recovery.shape[0] >= num_frames:
        gap = recovery[:num_frames]
        return gap, _tail_context_qpos(
            np.concatenate([context_qpos, gap], axis=0),
            demo.agent.NUM_FRAMES_PER_TOKEN,
        ), num_frames

    hold = np.repeat(recovery[-1:], num_frames - recovery.shape[0], axis=0)
    gap = np.concatenate([recovery, hold], axis=0)
    context_qpos = _tail_context_qpos(
        np.concatenate([context_qpos, gap], axis=0),
        demo.agent.NUM_FRAMES_PER_TOKEN,
    )
    return gap, context_qpos, recovery.shape[0]


def _reset_episode_context(demo: Demo) -> None:
    demo.agent.reset()
    first_qpos = _to_numpy(demo.agent.frames["mujoco_qpos"][0, 0]).astype(np.float64)
    demo.mj_data.qpos[:] = first_qpos
    mujoco.mj_forward(demo.mj_model, demo.mj_data)
    demo.controller.latest_root_target = None
    demo.controller.latest_hit_pos = None
    demo.controller.latest_hit_vel = None
    demo.controller.latest_racket_positions = None
    demo.controller.latest_keyframe_qpos = None
    demo.controller.latest_keyframe_qposes = None
    demo.controller.latest_mode_name = "frozen_default"


def _initial_episode_context(demo: Demo) -> np.ndarray:
    return _to_numpy(
        demo.agent.frames["mujoco_qpos"][0, :demo.agent.NUM_FRAMES_PER_TOKEN]
    ).astype(np.float32)


def _latest_ik_error(demo: Demo) -> float:
    hit_pos = demo.controller.latest_hit_pos
    hit_vel = demo.controller.latest_hit_vel
    racket_positions = demo.controller.latest_racket_positions
    offsets_s = demo.controller._target_frame_offsets / float(demo.controller.FPS)
    targets = np.asarray(hit_pos, dtype=np.float32)[None] + offsets_s[:, None] * np.asarray(hit_vel, dtype=np.float32)[None]
    return float(np.min(np.linalg.norm(np.asarray(racket_positions) - targets, axis=-1)))


def _effective_hit_time_s(demo: Demo) -> float:
    total_frames = int(demo.controller.latest_pred_num_tokens) * demo.agent.NUM_FRAMES_PER_TOKEN
    hit_frame = total_frames - demo.agent.NUM_FRAMES_PER_TOKEN - float(np.max(demo.controller._target_frame_offsets))
    return max(0.0, float(hit_frame) / float(demo.controller.FPS))


def _clip_metadata(demo: Demo, clip_index: int, start_frame: int, end_frame: int) -> dict[str, np.ndarray]:
    stroke_sign = 1 if demo.controller.latest_mode_name.endswith("forehand") else -1
    return {
        "clip_index": np.array([clip_index], dtype=np.int64),
        "command_start_frame": np.array([start_frame], dtype=np.int64),
        "command_end_frame": np.array([end_frame], dtype=np.int64),
        "stroke_sign": np.array([stroke_sign], dtype=np.int8),
        "hit_pos": np.asarray(demo.controller.latest_hit_pos, dtype=np.float32),
        "hit_vel": np.asarray(demo.controller.latest_hit_vel, dtype=np.float32),
        "root_target": np.asarray(demo.controller.latest_root_target, dtype=np.float32),
        "ik_error": np.array([_latest_ik_error(demo)], dtype=np.float32),
        "pred_num_tokens": np.array([int(demo.controller.latest_pred_num_tokens)], dtype=np.int64),
        "hit_time_s": np.array([_effective_hit_time_s(demo)], dtype=np.float32),
        "next_resample_time_s": np.array([float(demo.controller._command.next_resample_time_s)], dtype=np.float32),
    }


def _episode_metadata(command_records: list[dict[str, np.ndarray]], clip_index: int) -> dict[str, np.ndarray]:
    return {
        "clip_index": np.array([clip_index], dtype=np.int64),
        "command_start_frames": np.asarray(
            [r["command_start_frame"][0] for r in command_records], dtype=np.int64
        ),
        "command_end_frames": np.asarray(
            [r["command_end_frame"][0] for r in command_records], dtype=np.int64
        ),
        "stroke_sign": np.asarray([r["stroke_sign"][0] for r in command_records], dtype=np.int8),
        "hit_pos": np.stack([r["hit_pos"] for r in command_records]).astype(np.float32),
        "hit_vel": np.stack([r["hit_vel"] for r in command_records]).astype(np.float32),
        "root_target": np.stack([r["root_target"] for r in command_records]).astype(np.float32),
        "ik_error": np.asarray([r["ik_error"][0] for r in command_records], dtype=np.float32),
        "pred_num_tokens": np.asarray([r["pred_num_tokens"][0] for r in command_records], dtype=np.int64),
        "hit_time_s": np.asarray([r["hit_time_s"][0] for r in command_records], dtype=np.float32),
        "next_resample_time_s": np.asarray(
            [r["next_resample_time_s"][0] for r in command_records], dtype=np.float32
        ),
    }


def _frame_time_to_hit(
    command_records: list[dict[str, np.ndarray]],
    num_frames: int,
    fps: int,
) -> np.ndarray:
    values = np.zeros((num_frames,), dtype=np.float32)
    if not command_records:
        raise ValueError("No command records were generated.")
    starts = np.asarray([int(record["command_start_frame"][0]) for record in command_records], dtype=np.int64)
    ends = np.asarray([int(record["command_end_frame"][0]) for record in command_records], dtype=np.int64)
    hit_times = np.asarray([float(record["hit_time_s"][0]) for record in command_records], dtype=np.float32)
    for frame_idx in range(num_frames):
        cmd_idx = int(np.clip(np.searchsorted(starts, frame_idx, side="right") - 1, 0, len(starts) - 1))
        raw = hit_times[cmd_idx] - float(frame_idx - starts[cmd_idx]) / float(fps)
        motion_end_time_to_hit = hit_times[cmd_idx] - float(ends[cmd_idx] - starts[cmd_idx]) / float(fps)
        values[frame_idx] = max(float(motion_end_time_to_hit), raw)
    return values


def _reference_quality_metrics(payload: dict[str, np.ndarray]) -> dict[str, float]:
    metrics = {
        "max_joint_jump_rad": 0.0,
        "max_joint_vel_abs": 0.0,
        "max_body_lin_speed": 0.0,
        "max_body_ang_speed": 0.0,
        "nonfinite_count": 0.0,
    }
    for value in payload.values():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            metrics["nonfinite_count"] += float((~np.isfinite(arr)).sum())
    if "joint_pos" in payload:
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        if joint_pos.shape[0] > 1:
            metrics["max_joint_jump_rad"] = float(np.max(np.abs(np.diff(joint_pos, axis=0))))
    if "joint_vel" in payload:
        metrics["max_joint_vel_abs"] = float(np.max(np.abs(np.asarray(payload["joint_vel"], dtype=np.float64))))
    if "body_lin_vel_w" in payload:
        metrics["max_body_lin_speed"] = float(
            np.max(np.linalg.norm(np.asarray(payload["body_lin_vel_w"], dtype=np.float64), axis=-1)))
    if "body_ang_vel_w" in payload:
        metrics["max_body_ang_speed"] = float(
            np.max(np.linalg.norm(np.asarray(payload["body_ang_vel_w"], dtype=np.float64), axis=-1)))
    return metrics


def _reference_quality_reasons(
    payload: dict[str, np.ndarray],
    quality_cfg: dict,
) -> tuple[list[str], dict[str, float]]:
    if not bool(quality_cfg.get("enabled", False)):
        return [], {}
    metrics = _reference_quality_metrics(payload)
    reasons = []
    if metrics["nonfinite_count"] > 0:
        reasons.append(f"nonfinite={int(metrics['nonfinite_count'])}")
    checks = [
        ("max_joint_jump_rad", "max_joint_jump_rad"),
        ("max_joint_vel_abs", "max_joint_vel_rad_s"),
        ("max_body_lin_speed", "max_body_lin_speed_m_s"),
        ("max_body_ang_speed", "max_body_ang_speed_rad_s"),
    ]
    for metric_key, cfg_key in checks:
        limit = quality_cfg.get(cfg_key, None)
        if limit is not None and metrics[metric_key] > float(limit):
            reasons.append(f"{metric_key}={metrics[metric_key]:.4g}>{float(limit):.4g}")
    return reasons, metrics


def _atomic_save_npz(payload: dict[str, np.ndarray], pending_path: Path, ready_path: Path) -> None:
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pending_path, **payload)
    os.replace(pending_path, ready_path)


def _ready_files(ready_dir: Path) -> list[Path]:
    return sorted(ready_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)


def _prune_ready_dir(ready_dir: Path, max_ready_files: int) -> None:
    files = _ready_files(ready_dir)
    for path in files[: max(0, len(files) - max_ready_files)]:
        path.unlink(missing_ok=True)


def _make_room_for_next_ready_file(ready_dir: Path, max_ready_files: int) -> None:
    files = _ready_files(ready_dir)
    num_to_delete = max(0, len(files) - max_ready_files + 1)
    for path in files[:num_to_delete]:
        path.unlink(missing_ok=True)


def _generate_episode(
    demo: Demo,
    target_episode_frames: int,
    source_fps: int,
    controller_dt: float,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    _reset_episode_context(demo)
    segments: list[np.ndarray] = []
    records: list[dict[str, np.ndarray]] = []
    context_qpos = _initial_episode_context(demo)
    command_idx = 0

    while sum(segment.shape[0] for segment in segments) < target_episode_frames:
        current_num_frames = sum(segment.shape[0] for segment in segments)
        command_start_frame = current_num_frames

        full_hit_segment = generate_hit_segment_from_context(demo, context_qpos, controller_dt)
        target_next_frames = int(round(float(demo.controller._command.next_resample_time_s) * source_fps))
        remaining_frames = target_episode_frames - current_num_frames
        played_hit_frames = min(full_hit_segment.shape[0], max(1, target_next_frames), remaining_frames)

        hit_segment = full_hit_segment[:played_hit_frames]
        segments.append(hit_segment)
        hit_end_frame = sum(segment.shape[0] for segment in segments)
        command_end_frame = hit_end_frame
        context_qpos = _tail_context_qpos(
            np.concatenate([context_qpos, hit_segment], axis=0),
            demo.agent.NUM_FRAMES_PER_TOKEN,
        )

        if played_hit_frames >= full_hit_segment.shape[0]:
            gap_frames = target_next_frames - (hit_end_frame - command_start_frame)
            remaining_frames = target_episode_frames - sum(segment.shape[0] for segment in segments)
            gap_frames = min(max(0, gap_frames), max(0, remaining_frames))
            if gap_frames > 0:
                gap, context_qpos, recovery_frames = _generate_no_command_gap(
                    demo, gap_frames, controller_dt, context_qpos=context_qpos)
                command_end_frame = hit_end_frame + recovery_frames
                segments.append(gap)

        records.append(_clip_metadata(demo, command_idx, command_start_frame, command_end_frame))
        command_idx += 1

    if not segments:
        raise RuntimeError("Episode generation produced no motion segments.")
    return np.concatenate(segments, axis=0), records


def run(args) -> None:
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    buffer_cfg = OmegaConf.to_container(cfg["reference_buffer"], resolve=True)
    if args.output_dir is not None:
        buffer_cfg["output_dir"] = args.output_dir
    if args.num_clips is not None:
        buffer_cfg["num_clips"] = args.num_clips
    if args.continuous:
        buffer_cfg["continuous"] = True

    seed = buffer_cfg["seed"]
    if seed is not None:
        np.random.seed(int(seed))
        t.manual_seed(int(seed))

    output_dir = Path(buffer_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    pending_dir = output_dir / "pending"
    ready_dir = output_dir / "ready"
    pending_dir.mkdir(parents=True, exist_ok=True)
    ready_dir.mkdir(parents=True, exist_ok=True)

    demo_args = parse_demo_args(["--config", args.config, *args.overrides])
    demo = Demo(demo_args)
    source_fps = int(demo.inferencer.motion_rep.fps)
    output_fps = int(buffer_cfg["output_fps"])
    num_clips = int(buffer_cfg["num_clips"])
    episode_duration_s = float(buffer_cfg["episode_duration_s"])
    target_episode_frames = max(1, int(round(episode_duration_s * source_fps)))
    continuous = bool(buffer_cfg["continuous"])
    max_ready_files = int(buffer_cfg["max_ready_files"])
    quality_cfg = dict(buffer_cfg.get("quality_filter", {}))

    print(
        f"Writing MotionBricks ping-pong reference clips to {ready_dir} "
        f"(source_fps={source_fps}, output_fps={output_fps})"
    )
    clip_index = 0
    reject_count = 0
    controller_dt = demo.controller_dt
    while continuous or clip_index < num_clips:
        if continuous and len(_ready_files(ready_dir)) >= max_ready_files:
            _make_room_for_next_ready_file(ready_dir, max_ready_files)

        qpos, records = _generate_episode(demo, target_episode_frames, source_fps, controller_dt)
        output_qpos = _interpolate_resample_qpos(qpos, source_fps, output_fps)
        output_records = _rescale_command_records(records, source_fps, output_fps, output_qpos.shape[0])
        payload = qpos_to_isaaclab_motion_npz(
            demo.mj_model,
            output_qpos,
            output_fps,
            _episode_metadata(output_records, clip_index),
        )
        payload["source_fps"] = np.array([source_fps], dtype=np.int64)
        payload["resample_method"] = np.array(["linear_qpos_slerp_root_quat"])
        payload["time_to_hit_s"] = _frame_time_to_hit(output_records, output_qpos.shape[0], output_fps)
        quality_reasons, quality_metrics = _reference_quality_reasons(payload, quality_cfg)
        if quality_reasons:
            reject_count += 1
            print(
                f"reject quality #{reject_count}: "
                f"{'; '.join(quality_reasons)} "
                f"commands={len(records)} frames={output_qpos.shape[0]}"
            )
            continue
        stem = f"{clip_index:08d}_episode_{len(records)}cmd_{output_qpos.shape[0]}f_{output_fps}hz"
        _atomic_save_npz(payload, pending_dir / f"{stem}.npz", ready_dir / f"{stem}.npz")
        _prune_ready_dir(ready_dir, max_ready_files)
        print(
            f"ready {stem}: frames={output_qpos.shape[0]}, commands={len(records)}, "
            f"ik_mean={payload['ik_error'].mean():.4f}, "
            f"joint_jump={quality_metrics.get('max_joint_jump_rad', 0.0):.4f}, "
            f"joint_vel={quality_metrics.get('max_joint_vel_abs', 0.0):.2f}"
        )
        clip_index += 1


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/pingpong_g1.yaml"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_clips", type=int, default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
