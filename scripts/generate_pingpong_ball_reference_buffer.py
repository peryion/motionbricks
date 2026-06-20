#!/usr/bin/env python3
"""Generate MotionBricks ping-pong references from sampled incoming balls.

This is a small variant of generate_pingpong_reference_buffer.py:
sample an incoming ball, keep samples that reach the hit plane no earlier than
the fixed MotionBricks hit time, advance the ball to the active-motion start
state, then force the command controller to generate a matching hit + recovery
reference.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as t
from omegaconf import OmegaConf

from generate_pingpong_reference_buffer import (
    Demo,
    _atomic_save_npz,
    _frame_time_to_hit,
    _generate_no_command_gap,
    _initial_episode_context,
    _interpolate_resample_qpos,
    _latest_ik_error,
    _ready_files,
    _reset_episode_context,
    _rescale_command_records,
    _tail_context_qpos,
    generate_hit_segment_from_context,
    parse_demo_args,
    qpos_to_isaaclab_motion_npz,
)


@dataclass
class BallCommand:
    start_pos: np.ndarray
    start_vel: np.ndarray
    hit_pos: np.ndarray
    incoming_vel: np.ndarray
    hit_time_s: float
    motion_hit_vel: np.ndarray


def _uniform(rng: np.random.Generator, lo: float, hi: float, shape) -> np.ndarray:
    return rng.uniform(float(lo), float(hi), size=shape).astype(np.float32)


def _drag_step(vel: np.ndarray, dt: float, cfg: dict) -> np.ndarray:
    coeff = np.asarray(cfg["ball_air_drag_coeff_xyz"], dtype=np.float32).reshape(1, 3)
    mass = max(float(cfg["ball_mass"]), 1e-6)
    return mass * vel / (mass + coeff * np.abs(vel) * float(dt))


def _apply_table_bounce(
    last_pos: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_z = float(cfg["table_height"]) + float(cfg["ball_radius"])
    crossed = (last_pos[:, 2] > table_z) & (pos[:, 2] <= table_z) & (vel[:, 2] < 0.0)
    denom = np.minimum(pos[:, 2] - last_pos[:, 2], -1e-6)
    alpha = np.clip((table_z - last_pos[:, 2]) / denom, 0.0, 1.0)
    contact = last_pos + alpha[:, None] * (pos - last_pos)
    in_table = (
        (contact[:, 0] >= float(cfg["table_x_min"]))
        & (contact[:, 0] <= float(cfg["table_x_max"]))
        & (np.abs(contact[:, 1] - float(cfg["table_center_y"])) <= float(cfg["table_half_width"]))
    )
    hit = crossed & in_table
    if not np.any(hit):
        return pos, vel, hit
    out_pos = pos.copy()
    out_vel = vel.copy()
    k = np.asarray(cfg["table_bounce_k"], dtype=np.float32).reshape(1, 3)
    b_abs = np.asarray(cfg["table_bounce_b_abs"], dtype=np.float32).reshape(1, 3)
    out_pos[hit] = contact[hit]
    out_vel[hit] = k * vel[hit] + b_abs * np.sign(vel[hit])
    return out_pos, out_vel, hit


def _predict_hit_plane(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = float(cfg["sim_dt"])
    steps = max(1, math.ceil(float(cfg["max_time_to_hit_s"]) / dt))
    gravity = np.array([0.0, 0.0, -float(cfg["ball_gravity"])], dtype=np.float32)
    pos = start_pos.copy()
    vel = start_vel.copy()
    hit_pos = np.zeros_like(pos)
    hit_vel = np.zeros_like(vel)
    hit_time = np.full((pos.shape[0],), float(cfg["max_time_to_hit_s"]), dtype=np.float32)
    found = np.zeros(pos.shape[0], dtype=bool)
    bounce_count = np.zeros(pos.shape[0], dtype=np.int64)
    hit_x = float(cfg["hit_plane_x"])
    table_z = float(cfg["table_height"]) + float(cfg["ball_radius"])
    own_table_x_max = min(float(cfg["table_x_max"]), float(cfg["net_x"]))

    for step in range(steps):
        prev_pos = pos.copy()
        prev_vel = vel.copy()
        vel = _drag_step(vel, dt, cfg)
        vel = vel + gravity[None, :] * dt
        pos = pos + vel * dt

        before_bounce_pos = pos.copy()
        pos, vel, table_hit = _apply_table_bounce(prev_pos, pos, vel, cfg)
        if np.any(table_hit):
            contact = pos
            in_own_table = (
                (contact[:, 0] >= float(cfg["table_x_min"]))
                & (contact[:, 0] <= own_table_x_max)
                & (np.abs(contact[:, 1] - float(cfg["table_center_y"])) <= float(cfg["table_half_width"]))
                & (np.abs(contact[:, 2] - table_z) < 1e-3)
            )
            bounce_count += (table_hit & in_own_table & (~found)).astype(np.int64)
            pos = np.where(table_hit[:, None], pos, before_bounce_pos)

        crossed = (~found) & (prev_pos[:, 0] >= hit_x) & (pos[:, 0] <= hit_x)
        if np.any(crossed):
            ids = np.flatnonzero(crossed)
            denom = np.maximum(prev_pos[ids, 0] - pos[ids, 0], 1e-6)
            alpha = np.clip((prev_pos[ids, 0] - hit_x) / denom, 0.0, 1.0)
            hit_pos[ids] = prev_pos[ids] + alpha[:, None] * (pos[ids] - prev_pos[ids])
            hit_pos[ids, 0] = hit_x
            hit_vel[ids] = prev_vel[ids] + alpha[:, None] * (vel[ids] - prev_vel[ids])
            hit_time[ids] = (float(step) + alpha) * dt
            found[ids] = True

    return hit_time, hit_pos, hit_vel, bounce_count, found


def _advance_ball_state(pos: np.ndarray, vel: np.ndarray, duration_s: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32).reshape(1, 3).copy()
    vel = np.asarray(vel, dtype=np.float32).reshape(1, 3).copy()
    gravity = np.array([0.0, 0.0, -float(cfg["ball_gravity"])], dtype=np.float32)
    remain = max(0.0, float(duration_s))
    sim_dt = float(cfg["sim_dt"])
    while remain > 1e-8:
        dt = min(sim_dt, remain)
        last_pos = pos.copy()
        vel = _drag_step(vel, dt, cfg)
        vel = vel + gravity[None, :] * dt
        pos = pos + vel * dt
        pos, vel, _ = _apply_table_bounce(last_pos, pos, vel, cfg)
        remain -= dt
    return pos[0].astype(np.float32), vel[0].astype(np.float32)


def _sample_ball_command(rng: np.random.Generator, cfg: dict, target_hit_time_s: float) -> BallCommand:
    batch = int(cfg["sample_batch_size"])
    for _ in range(int(cfg["sample_max_batches"])):
        start_pos = np.zeros((batch, 3), dtype=np.float32)
        start_pos[:, 0] = _uniform(rng, *cfg["ball_start_x_range"], (batch,))
        start_pos[:, 1] = _uniform(rng, *cfg["ball_start_y_range"], (batch,))
        start_pos[:, 2] = _uniform(rng, *cfg["ball_start_z_range"], (batch,))
        start_vel = np.zeros((batch, 3), dtype=np.float32)
        start_vel[:, 0] = _uniform(rng, *cfg["ball_incoming_vx_range"], (batch,))
        start_vel[:, 1] = _uniform(rng, *cfg["ball_incoming_vy_range"], (batch,))
        start_vel[:, 2] = _uniform(rng, *cfg["ball_incoming_vz_range"], (batch,))

        hit_time, hit_pos, incoming_vel, bounce_count, found = _predict_hit_plane(start_pos, start_vel, cfg)
        valid = (
            found
            & (bounce_count == int(cfg["ball_required_bounce_count_before_hit"]))
            & (hit_time >= float(target_hit_time_s))
            & (hit_pos[:, 1] >= float(cfg["hit_y_range"][0]))
            & (hit_pos[:, 1] <= float(cfg["hit_y_range"][1]))
            & (hit_pos[:, 2] >= float(cfg["hit_z_range"][0]))
            & (hit_pos[:, 2] <= float(cfg["hit_z_range"][1]))
            & np.isfinite(hit_pos).all(axis=1)
            & np.isfinite(incoming_vel).all(axis=1)
        )
        ids = np.flatnonzero(valid)
        if ids.size == 0:
            continue
        idx = int(rng.choice(ids))
        motion_hit_vel = np.asarray(cfg["motion_hit_vel"], dtype=np.float32).copy()
        motion_hit_vel[1] += float(cfg["motion_hit_vel_y_incoming_gain"]) * float(-incoming_vel[idx, 1])
        motion_hit_vel[2] += float(cfg["motion_hit_vel_z_incoming_gain"]) * float(incoming_vel[idx, 2])
        motion_hit_vel[1] = np.clip(motion_hit_vel[1], -0.35, 0.35)
        motion_hit_vel[2] = np.clip(motion_hit_vel[2], -0.05, 0.45)
        wait_s = float(hit_time[idx]) - float(target_hit_time_s)
        command_start_pos, command_start_vel = _advance_ball_state(start_pos[idx], start_vel[idx], wait_s, cfg)
        return BallCommand(
            start_pos=command_start_pos,
            start_vel=command_start_vel,
            hit_pos=hit_pos[idx],
            incoming_vel=incoming_vel[idx],
            hit_time_s=float(target_hit_time_s),
            motion_hit_vel=motion_hit_vel,
        )
    raise RuntimeError("Could not sample a valid ball command. Widen ranges or hit_time_tolerance_s.")


def _with_external_command(demo: Demo, ball_cmd: BallCommand, next_resample_time_s: float, fn):
    command = demo.controller._command
    old_resample = command.resample

    def resample(base_y: float = 0.0):
        command.elapsed_s = 0.0
        command.command_id += 1
        command.time_to_hit_frozen = False
        command.next_resample_time_s = float(next_resample_time_s)
        command.target_hit_pos[:] = ball_cmd.hit_pos.astype(np.float32)
        command.target_hit_vel[:] = ball_cmd.motion_hit_vel.astype(np.float32)
        split_y = float(base_y) + float(command.cfg.get("mixed_backhand_split_offset_y", -0.2))
        command.is_forehand = bool(ball_cmd.hit_pos[1] <= split_y)
        command.target_root_xy[:] = np.array(
            [float(command.cfg.get("root_target_x", 0.0)), float(ball_cmd.hit_pos[1])],
            dtype=np.float32,
        )

    command.resample = resample
    try:
        return fn()
    finally:
        command.resample = old_resample


def _record(demo: Demo, ball_cmd: BallCommand, clip_index: int, start_frame: int, end_frame: int) -> dict[str, np.ndarray]:
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
        "hit_time_s": np.array([float(ball_cmd.hit_time_s)], dtype=np.float32),
        "next_resample_time_s": np.array([float(demo.controller._command.next_resample_time_s)], dtype=np.float32),
        "ball_start_pos": ball_cmd.start_pos.astype(np.float32),
        "ball_start_vel": ball_cmd.start_vel.astype(np.float32),
        "ball_hit_pos": ball_cmd.hit_pos.astype(np.float32),
        "ball_hit_vel": ball_cmd.incoming_vel.astype(np.float32),
        "ball_hit_time_s": np.array([float(ball_cmd.hit_time_s)], dtype=np.float32),
        "motion_hit_vel": ball_cmd.motion_hit_vel.astype(np.float32),
    }


def _episode_metadata(records: list[dict[str, np.ndarray]], clip_index: int) -> dict[str, np.ndarray]:
    out = {"clip_index": np.array([clip_index], dtype=np.int64)}
    out["command_start_frames"] = np.asarray([r["command_start_frame"][0] for r in records], dtype=np.int64)
    out["command_end_frames"] = np.asarray([r["command_end_frame"][0] for r in records], dtype=np.int64)
    out["stroke_sign"] = np.asarray([r["stroke_sign"][0] for r in records], dtype=np.int8)
    out["ik_error"] = np.asarray([r["ik_error"][0] for r in records], dtype=np.float32)
    out["pred_num_tokens"] = np.asarray([r["pred_num_tokens"][0] for r in records], dtype=np.int64)
    out["hit_time_s"] = np.asarray([r["hit_time_s"][0] for r in records], dtype=np.float32)
    out["next_resample_time_s"] = np.asarray([r["next_resample_time_s"][0] for r in records], dtype=np.float32)
    out["ball_hit_time_s"] = np.asarray([r["ball_hit_time_s"][0] for r in records], dtype=np.float32)
    out["hit_pos"] = np.stack([r["hit_pos"] for r in records]).astype(np.float32)
    out["hit_vel"] = np.stack([r["hit_vel"] for r in records]).astype(np.float32)
    out["root_target"] = np.stack([r["root_target"] for r in records]).astype(np.float32)
    out["ball_start_pos"] = np.stack([r["ball_start_pos"] for r in records]).astype(np.float32)
    out["ball_start_vel"] = np.stack([r["ball_start_vel"] for r in records]).astype(np.float32)
    out["ball_hit_pos"] = np.stack([r["ball_hit_pos"] for r in records]).astype(np.float32)
    out["ball_hit_vel"] = np.stack([r["ball_hit_vel"] for r in records]).astype(np.float32)
    out["motion_hit_vel"] = np.stack([r["motion_hit_vel"] for r in records]).astype(np.float32)
    return out


def _frame_ball_state(records: list[dict[str, np.ndarray]], num_frames: int, fps: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((num_frames, 3), dtype=np.float32)
    vel = np.zeros((num_frames, 3), dtype=np.float32)
    starts = [int(r["command_start_frame"][0]) for r in records]
    for frame in range(num_frames):
        cmd_idx = int(np.clip(np.searchsorted(starts, frame, side="right") - 1, 0, len(records) - 1))
        rec = records[cmd_idx]
        elapsed = max(0.0, float(frame - int(rec["command_start_frame"][0])) / float(fps))
        p, v = rec["ball_start_pos"].copy(), rec["ball_start_vel"].copy()
        steps = max(0, int(round(elapsed / float(cfg["sim_dt"]))))
        for _ in range(steps):
            last = p[None, :].copy()
            v = _drag_step(v[None, :], float(cfg["sim_dt"]), cfg)[0]
            v = v + np.array([0.0, 0.0, -float(cfg["ball_gravity"])], dtype=np.float32) * float(cfg["sim_dt"])
            p = p + v * float(cfg["sim_dt"])
            p, v, _ = _apply_table_bounce(last, p[None, :], v[None, :], cfg)
            p, v = p[0], v[0]
        pos[frame], vel[frame] = p, v
    return pos, vel


def _generate_episode(demo: Demo, target_episode_frames: int, source_fps: int, controller_dt: float, cfg: dict, rng: np.random.Generator):
    _reset_episode_context(demo)
    segments: list[np.ndarray] = []
    records: list[dict[str, np.ndarray]] = []
    context_qpos = _initial_episode_context(demo)
    motion_hit_time_s = demo.controller._hit_time_from_pred_tokens(demo.controller._fixed_pred_num_tokens())

    while sum(s.shape[0] for s in segments) < target_episode_frames:
        start_frame = sum(s.shape[0] for s in segments)
        ball_cmd = _sample_ball_command(rng, cfg, motion_hit_time_s)
        next_resample_time_s = float(rng.uniform(*cfg["next_resample_time_range_s"]))

        def make_hit():
            return generate_hit_segment_from_context(demo, context_qpos, controller_dt)

        full_hit = _with_external_command(demo, ball_cmd, next_resample_time_s, make_hit)
        target_next_frames = int(round(next_resample_time_s * source_fps))
        remaining = target_episode_frames - start_frame
        played = min(full_hit.shape[0], max(1, target_next_frames), remaining)
        hit_segment = full_hit[:played]
        segments.append(hit_segment)
        hit_end_frame = sum(s.shape[0] for s in segments)
        command_end_frame = hit_end_frame
        context_qpos = _tail_context_qpos(np.concatenate([context_qpos, hit_segment], axis=0), demo.agent.NUM_FRAMES_PER_TOKEN)

        if played >= full_hit.shape[0]:
            gap_frames = min(max(0, target_next_frames - (hit_end_frame - start_frame)), target_episode_frames - hit_end_frame)
            if gap_frames > 0:
                gap, context_qpos, recovery_frames = _generate_no_command_gap(
                    demo, gap_frames, controller_dt, context_qpos=context_qpos
                )
                command_end_frame = hit_end_frame + recovery_frames
                segments.append(gap)

        records.append(_record(demo, ball_cmd, len(records), start_frame, command_end_frame))
    return np.concatenate(segments, axis=0), records


def _default_ball_cfg() -> dict:
    return {
        "sim_dt": 0.02,
        "ball_radius": 0.02,
        "ball_mass": 0.0027,
        "ball_gravity": 9.81,
        "ball_air_drag_coeff_xyz": (0.0003, 0.0008, 0.0005),
        "ball_start_x_range": (1.67, 3.24),
        "ball_start_y_range": (-0.76, 0.76),
        "ball_start_z_range": (0.90, 1.50),
        "ball_incoming_vx_range": (-10.0, -1.0),
        "ball_incoming_vy_range": (-1.0, 1.0),
        "ball_incoming_vz_range": (-2.0, 4.0),
        "ball_required_bounce_count_before_hit": 1,
        "max_time_to_hit_s": 0.8,
        "hit_plane_x": 0.3,
        "hit_y_range": (-0.76, 0.76),
        "hit_z_range": (0.76, 1.26),
        "hit_time_tolerance_s": 0.04,
        "table_height": 0.76,
        "table_x_min": 0.50,
        "table_x_max": 3.24,
        "table_center_y": 0.0,
        "table_half_width": 0.76,
        "table_bounce_k": (0.50, 0.75, -0.89),
        "table_bounce_b_abs": (0.51, 0.0, 0.0),
        "net_x": 1.87,
        "motion_hit_vel": (1.6, 0.0, 0.2),
        "motion_hit_vel_y_incoming_gain": 0.15,
        "motion_hit_vel_z_incoming_gain": 0.05,
        "next_resample_time_range_s": (1.0, 1.5),
        "sample_batch_size": 4096,
        "sample_max_batches": 32,
    }


def run(args) -> None:
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    buffer_cfg = OmegaConf.to_container(cfg["reference_buffer"], resolve=True)
    if args.output_dir is not None:
        buffer_cfg["output_dir"] = args.output_dir
    if args.num_clips is not None:
        buffer_cfg["num_clips"] = args.num_clips

    ball_cfg = _default_ball_cfg()
    ball_override = cfg.get("ball_reference_buffer", None)
    if ball_override is not None:
        ball_cfg.update(OmegaConf.to_container(ball_override, resolve=True) or {})
    if args.hit_time_tolerance_s is not None:
        ball_cfg["hit_time_tolerance_s"] = args.hit_time_tolerance_s

    seed = buffer_cfg.get("seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))
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
    target_episode_frames = max(1, int(round(float(buffer_cfg["episode_duration_s"]) * source_fps)))
    num_clips = int(buffer_cfg["num_clips"])
    controller_dt = demo.controller_dt
    print(f"Writing ball-driven references to {ready_dir} (source_fps={source_fps}, output_fps={output_fps})")

    for clip_index in range(num_clips):
        qpos, records = _generate_episode(demo, target_episode_frames, source_fps, controller_dt, ball_cfg, rng)
        out_qpos = _interpolate_resample_qpos(qpos, source_fps, output_fps)
        out_records = _rescale_command_records(records, source_fps, output_fps, out_qpos.shape[0])
        metadata = _episode_metadata(out_records, clip_index)
        payload = qpos_to_isaaclab_motion_npz(demo.mj_model, out_qpos, output_fps, metadata)
        payload["source_fps"] = np.array([source_fps], dtype=np.int64)
        payload["resample_method"] = np.array(["linear_qpos_slerp_root_quat"])
        payload["time_to_hit_s"] = _frame_time_to_hit(out_records, out_qpos.shape[0], output_fps)
        payload["ball_pos_w"], payload["ball_vel_w"] = _frame_ball_state(out_records, out_qpos.shape[0], output_fps, ball_cfg)
        stem = f"{clip_index:08d}_ball_episode_{len(records)}cmd_{out_qpos.shape[0]}f_{output_fps}hz"
        _atomic_save_npz(payload, pending_dir / f"{stem}.npz", ready_dir / f"{stem}.npz")
        max_ready = int(buffer_cfg.get("max_ready_files", num_clips))
        for old in _ready_files(ready_dir)[: max(0, len(_ready_files(ready_dir)) - max_ready)]:
            old.unlink(missing_ok=True)
        print(
            f"ready {stem}: commands={len(records)}, "
            f"ball_t={payload['ball_hit_time_s'].mean():.3f}s, "
            f"ik={payload['ik_error'].mean():.4f}"
        )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/pingpong_g1.yaml"))
    parser.add_argument("--output_dir", default="out/reference_ball/pingpong_fk")
    parser.add_argument("--num_clips", type=int, default=None)
    parser.add_argument("--hit_time_tolerance_s", type=float, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
