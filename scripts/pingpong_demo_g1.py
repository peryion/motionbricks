#!/usr/bin/env python3
"""Target-pose demo for ping-pong commands.

Press N to trigger a hit:
  1. IK finds arm joint angles to reach the sampled ball target.
  2. A hit motion is generated, conditioned on those 4 target-pose frames.
  3. After the hit motion plays out, a recovery (frozen-default) motion is
     automatically generated and played before returning to idle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import os
import time
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import torch as t

from pingpong_command import PingPongCommandState
from motionbricks.exp_setup.experiment import test
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from pingpong_motion_agent import PingPongMotionAgent, force_generate_and_trim

from pingpong_demo_utils import (
    KeyboardState as ViewerKeyboardState,
    _load_model_from_training_config,
    _resolve_training_config_for_ckpt,
    build_mj_simulator,
    disable_mujoco_keyboard_shortcuts,
    draw_model_target_roots,
    draw_pingpong_table,
    draw_root_target,
)


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

class SafeKeyboardState:
    def __init__(self):
        try:
            self._impl = ViewerKeyboardState()
        except Exception as exc:
            print(f"Note: keyboard listener disabled: {exc}")
            self._impl = None

    def is_n_pressed(self) -> bool:
        if self._impl is None:
            return False
        pressed = getattr(self._impl, "_pressed", None)
        kb = getattr(self._impl, "_keyboard", None)
        return ("n" in pressed) if pressed is not None else (kb.is_pressed("n") if kb else False)


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

class RacketCenterIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        iterations: int = 30,
        damping: float = 1e-3,
        orientation_weight: float = 0.45,
        max_root_x: float | None = 0.25,
        max_root_yaw_delta: float = 0.35,
        max_waist_yaw: float = 0.35,
    ):
        self._model = model
        self._data = mujoco.MjData(model)
        self._iterations = iterations
        self._damping = damping
        self._orientation_weight = orientation_weight
        self._max_root_x = max_root_x
        self._max_root_yaw_delta = max_root_yaw_delta
        self._max_waist_yaw = max_waist_yaw
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"{site_name} site not found in MuJoCo model")
        joint_names = [
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self._joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names],
            dtype=np.int32)
        if np.any(self._joint_ids < 0):
            missing = [n for n, jid in zip(joint_names, self._joint_ids) if jid < 0]
            raise ValueError(f"Missing joints: {missing}")
        self._qadr = model.jnt_qposadr[self._joint_ids].copy()
        self._dadr = model.jnt_dofadr[self._joint_ids].copy()
        self._waist_yaw_qadr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")]

    def _racket_world_pos(self):
        return self._data.site_xpos[self._site_id].copy()

    def _racket_y_axis(self):
        return self._data.site_xmat[self._site_id].reshape(3, 3)[:, 1].copy()

    def forward_site_pos(self, qpos: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)
        return self._racket_world_pos()

    def forward_site_y_axis(self, qpos: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)
        return self._racket_y_axis()

    def _seed_arm_posture(self, qpos: np.ndarray, want_forehand: bool, seed_variant: int = 0):
        values = {
            "waist_yaw_joint": float(np.clip(
                -0.15 if want_forehand else 0.20, -self._max_waist_yaw, self._max_waist_yaw)),
            "right_shoulder_pitch_joint": -0.35,
            "right_shoulder_roll_joint": -0.55 if want_forehand else -0.10,
            "right_shoulder_yaw_joint": 0.25 if want_forehand else -0.35,
            "right_elbow_joint": 0.85,
            "right_wrist_roll_joint": 0.15 if want_forehand else -0.35,
            "right_wrist_pitch_joint": -0.15,
            "right_wrist_yaw_joint": 0.05 if want_forehand else -0.45,
        }
        variants = [
            {},
            {
                "right_shoulder_roll_joint": 0.12,
                "right_shoulder_yaw_joint": 0.16,
                "right_elbow_joint": -0.12,
                "right_wrist_yaw_joint": 0.22,
            },
            {
                "right_shoulder_roll_joint": -0.12,
                "right_shoulder_yaw_joint": -0.16,
                "right_elbow_joint": 0.12,
                "right_wrist_roll_joint": -0.18,
            },
            {
                "waist_yaw_joint": -0.08 if want_forehand else 0.08,
                "right_shoulder_pitch_joint": 0.12,
                "right_wrist_yaw_joint": -0.22,
            },
        ]
        for name, delta in variants[int(seed_variant) % len(variants)].items():
            values[name] = values.get(name, 0.0) + float(delta)
        for name, value in values.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos[self._model.jnt_qposadr[jid]] = value

    def _clip_joint_ranges(self, qpos: np.ndarray):
        for jid, qadr in zip(self._joint_ids, self._qadr):
            if self._model.jnt_limited[jid]:
                lo, hi = self._model.jnt_range[jid]
                qpos[qadr] = np.clip(qpos[qadr], lo, hi)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = v
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)

    @staticmethod
    def _safe_quat(qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos[3:7], dtype=np.float64).copy()
        n = np.linalg.norm(q)
        return q / n if (np.isfinite(n) and n > 1e-6) else np.array([1., 0., 0., 0.])

    def solve(self, seed_qpos: np.ndarray, palm_target: np.ndarray,
              target_y_axis: np.ndarray, want_forehand: bool,
              lock_root: bool = False,
              apply_seed_posture: bool = True,
              seed_variant: int = 0) -> np.ndarray:
        qpos = seed_qpos.copy()
        qpos[3:7] = self._safe_quat(qpos)
        if apply_seed_posture:
            self._seed_arm_posture(qpos, want_forehand, seed_variant=seed_variant)
        self._clip_joint_ranges(qpos)

        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        active_dofs = self._dadr if lock_root else np.concatenate(
            [np.array([0, 1, 5], dtype=np.int32), self._dadr])
        eye = np.eye(len(active_dofs))
        seed_root_z = float(qpos[2])
        seed_root_rot = R.from_quat(qpos[3:7], scalar_first=True)
        root_yaw_delta = 0.0
        root_dof_offset = 0 if lock_root else 3
        target_y_axis = target_y_axis / (np.linalg.norm(target_y_axis) + 1e-8)

        for _ in range(self._iterations):
            self._data.qpos[:] = qpos
            mujoco.mj_forward(self._model, self._data)
            pos_err = palm_target - self._racket_world_pos()
            axis = self._racket_y_axis()
            axis_err = target_y_axis - axis
            if np.linalg.norm(pos_err) < 0.006 and np.linalg.norm(axis_err) < 0.04:
                break
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
            J = np.concatenate(
                [jacp[:, active_dofs],
                 self._orientation_weight * (-self._skew(axis) @ jacr[:, active_dofs])], axis=0)
            rhs = J.T @ np.concatenate([pos_err, self._orientation_weight * axis_err])
            dq = np.linalg.solve(J.T @ J + self._damping * eye, rhs)

            if not lock_root:
                qpos[0] = min(qpos[0] + float(np.clip(dq[0], -0.08, 0.08)),
                              self._max_root_x if self._max_root_x is not None else 1e9)
                qpos[1] += float(np.clip(dq[1], -0.08, 0.08))
                root_yaw_delta = float(np.clip(
                    root_yaw_delta + float(np.clip(dq[2], -0.06, 0.06)),
                    -self._max_root_yaw_delta, self._max_root_yaw_delta))
                qpos[3:7] = (R.from_euler("z", root_yaw_delta) * seed_root_rot).as_quat(scalar_first=True)
            qpos[2] = seed_root_z
            qpos[self._qadr] += np.clip(dq[root_dof_offset:], -0.16, 0.16)
            qpos[self._waist_yaw_qadr] = float(np.clip(
                qpos[self._waist_yaw_qadr], -self._max_waist_yaw, self._max_waist_yaw))
            self._clip_joint_ranges(qpos)

        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)
        if not lock_root:
            residual = palm_target - self._racket_world_pos()
            qpos[:2] += residual[:2]
            if self._max_root_x is not None:
                qpos[0] = min(qpos[0], self._max_root_x)
        qpos[2] = seed_root_z
        return qpos


# ---------------------------------------------------------------------------
# command controller (IK sampling + command state, no keyboard)
# ---------------------------------------------------------------------------

@dataclass
class IKAttemptResult:
    target_qpos: np.ndarray
    axis_err: float
    backhand_root_ok: bool
    backhand_root_left_distance: float | None
    root_target: np.ndarray
    hit_pos: np.ndarray
    hit_vel: np.ndarray
    racket_positions: np.ndarray
    keyframe_qposes: np.ndarray
    keyframe_qpos: np.ndarray
    pred_num_tokens: int
    mode_name: str


class PingPongTargetController:
    """Samples hit commands and solves IK to produce target qpos for the agent.

    Does not handle keyboard or playback state — those live in the main loop.
    """

    NUM_TARGET_FRAMES = 4
    FPS = 30

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        min_token: int,
        max_token: int,
        allowed_min_token: int,
        allowed_max_token: int,
        default_pred_num_tokens: int,
        command_cfg: dict,
        target_heading: float = 0.0,
        default_qpos = None,
        ik_iterations: int = 30,
        ik_damping: float = 1e-3,
        racket_site_name: str = "paddle_center",
        target_frame_offsets: list[float] | tuple[float, ...] = (-1.5, -0.5, 0.5, 1.5),
        target_pose_mask: list[bool] | tuple[bool, ...] = (False, True, True, False),
        constrain_target_root: bool = False,
        orientation_weight: float = 0.45,
        max_axis_error_deg: float = 35.0,
        resample_attempts: int = 8,
        max_root_x: float | None = 0.25,
        max_root_yaw_delta: float = 0.35,
        max_waist_yaw: float = 0.35,
        max_backhand_root_left_distance_m: float | None = 0.45,
    ):
        self._min_token = min_token
        self._max_token = max_token
        self._allowed_min_token = allowed_min_token
        self._allowed_max_token = allowed_max_token
        self._default_pred_num_tokens = default_pred_num_tokens
        self._target_heading = target_heading
        self._default_qpos = default_qpos
        self._command = PingPongCommandState(command_cfg)
        self._target_frame_offsets = np.asarray(target_frame_offsets, dtype=np.float64)
        self._target_pose_mask = np.asarray(target_pose_mask, dtype=bool)
        if self._target_frame_offsets.shape[0] != self.NUM_TARGET_FRAMES:
            raise ValueError(f"target_frame_offsets must have {self.NUM_TARGET_FRAMES} values.")
        if self._target_pose_mask.shape[0] != self.NUM_TARGET_FRAMES:
            raise ValueError(f"target_pose_mask must have {self.NUM_TARGET_FRAMES} values.")
        if not self._target_pose_mask.any():
            raise ValueError("target_pose_mask must enable at least one frame.")
        self._constrain_target_root = constrain_target_root
        self._ik = RacketCenterIK(
            mj_model, site_name=racket_site_name, iterations=ik_iterations,
            damping=ik_damping, orientation_weight=orientation_weight,
            max_root_x=max_root_x, max_root_yaw_delta=max_root_yaw_delta,
            max_waist_yaw=max_waist_yaw)
        self._max_axis_error_deg = float(max_axis_error_deg)
        self._resample_attempts = max(1, int(resample_attempts))
        self._max_backhand_root_left_distance_m = (
            None if max_backhand_root_left_distance_m is None
            else float(max_backhand_root_left_distance_m)
        )
        # visual state (updated by sample_hit_control)
        self.latest_root_target: np.ndarray | None = None
        self.latest_hit_pos: np.ndarray | None = None
        self.latest_hit_vel: np.ndarray | None = None
        self.latest_racket_positions: np.ndarray | None = None
        self.latest_keyframe_qpos: np.ndarray | None = None
        self.latest_keyframe_qposes: np.ndarray | None = None
        self.latest_pred_num_tokens = default_pred_num_tokens
        self.latest_mode_name = "frozen_default"
        # pred_num_tokens is fixed, so the model's allowed-tokens 1-hot is constant
        lo = max(self._min_token, self._allowed_min_token)
        hi = min(self._max_token, self._allowed_max_token)
        token = int(np.clip(default_pred_num_tokens, lo, hi))
        allowed = t.zeros(self._max_token - self._min_token + 1, dtype=t.int)
        allowed[token - self._min_token] = 1
        self._allowed_pred_num_tokens = allowed.view(1, -1)

    def _fixed_pred_num_tokens(self) -> int:
        return int(np.clip(
            self._default_pred_num_tokens,
            max(self._min_token, self._allowed_min_token),
            min(self._max_token, self._allowed_max_token)))

    def _hit_time_from_pred_tokens(self, pred_tokens: int) -> float:
        hit_frame = (
            int(pred_tokens) * self.NUM_TARGET_FRAMES
            - self.NUM_TARGET_FRAMES
            - float(np.max(self._target_frame_offsets))
        )
        return max(0.0, hit_frame / float(self.FPS))

    # --- public API ----------------------------------------------------------

    def sample_hit_control(self, current_qpos: np.ndarray, command=None) -> dict:
        """Run IK, update visual state, and return a control dict for hit generation.

        If ``command`` is None (training/demo path), sample a fresh hit target
        via the internal command state. If ``command`` is provided (deploy path,
        either externally received or locally sampled by the caller), use its
        ``is_forehand`` / ``target_hit_pos`` / ``target_hit_vel`` directly. The
        IK-side hit timing is always derived from ``pred_num_tokens``; the
        external command's ``hit_time_s`` (if any) is the caller's concern.
        """
        target_qpos = self._build_target_qpos(current_qpos, external_command=command)
        mask = self._pose_mask_tensor()
        control = self._base_control()
        control.update({
            "target_mujoco_qpos": t.from_numpy(target_qpos).view(1, self.NUM_TARGET_FRAMES, -1).float(),
            "has_specific_target": t.ones(1, 1, dtype=t.int),
            "target_pose_mask": mask,
            "target_global_root_mask": (
                mask.clone() if self._constrain_target_root
                else t.zeros(1, self.NUM_TARGET_FRAMES, dtype=t.bool)),
            "target_local_root_mask": t.zeros(1, self.NUM_TARGET_FRAMES, dtype=t.bool),
        })
        return control

    def recovery_control(self, root_target: np.ndarray | None = None) -> dict:
        """Control dict for recovery / idle (frozen-default mode)."""
        control = self._base_control()
        target_qpos = np.repeat(self._default_qpos[None], self.NUM_TARGET_FRAMES, axis=0)
        if root_target is not None:
            target_qpos[:, :3] = np.asarray(root_target, dtype=np.float32)[None]
        target_qpos[:, 3:7] = np.array(
            [np.cos(self._target_heading * 0.5), 0.0, 0.0, np.sin(self._target_heading * 0.5)],
            dtype=np.float32,
        )
        control["target_mujoco_qpos"] = (
            t.from_numpy(target_qpos).view(1, self.NUM_TARGET_FRAMES, -1).float()
        )
        self.latest_root_target = target_qpos[-1, :3].astype(np.float32)
        return control

    # --- helpers -------------------------------------------------------------

    def _facing(self) -> t.Tensor:
        return t.tensor(
            [[np.cos(self._target_heading), np.sin(self._target_heading), 0.0]]).float()

    def _base_control(self) -> dict:
        return {
            "movement_direction": t.zeros(1, 3),
            "facing_direction": self._facing(),
            "allowed_pred_num_tokens": self._allowed_pred_num_tokens,
        }

    def _pose_mask_tensor(self) -> t.Tensor:
        mask = t.zeros(1, self.NUM_TARGET_FRAMES, dtype=t.bool)
        for idx in np.flatnonzero(self._target_pose_mask).tolist():
            mask[0, idx] = True
        return mask

    # --- IK sampling ---------------------------------------------------------

    def _apply_external_command(self, command) -> None:
        """Overwrite the internal command state with an external (caller-built)
        command. Expects fields: is_forehand, target_hit_pos, target_hit_vel.
        Bumps command_id and resets timing the same way resample() does so the
        rest of the controller flow is consistent.
        """
        self._command.command_id += 1
        self._command.elapsed_s = 0.0
        self._command.time_to_hit_frozen = False
        self._command.is_forehand = bool(command.is_forehand)
        self._command.target_hit_pos[:] = np.asarray(command.target_hit_pos, dtype=np.float32)
        self._command.target_hit_vel[:] = np.asarray(command.target_hit_vel, dtype=np.float32)
        root_target_x = float(self._command.cfg.get("root_target_x", 0.0))
        self._command.target_root_xy[:] = np.array(
            [root_target_x, float(command.target_hit_pos[1])], dtype=np.float32,
        )

    def _build_target_qpos(
        self, seed_qpos: np.ndarray, external_command=None,
    ) -> np.ndarray:
        """Sample (or accept) one command, try IK seeds, return best attempt."""
        if external_command is None:
            self._command.resample(base_y=float(seed_qpos[1]))
        else:
            self._apply_external_command(external_command)
        self._command.hit_time_s = self._hit_time_from_pred_tokens(self._fixed_pred_num_tokens())
        best_result, best_score = None, float("inf")
        for attempt in range(1, self._resample_attempts + 1):
            result = self._ik_attempt(seed_qpos, seed_variant=attempt - 1)
            score = result.axis_err + (0.0 if result.backhand_root_ok else 1000.0)
            if score < best_score or best_result is None:
                best_result = result
                best_score = score
            if result.axis_err <= self._max_axis_error_deg and result.backhand_root_ok:
                self._apply_ik_result(result)
                return result.target_qpos
            reasons = []
            if result.axis_err > self._max_axis_error_deg:
                reasons.append(f"axis error {result.axis_err:.1f}° > {self._max_axis_error_deg:.1f}°")
            if not result.backhand_root_ok:
                reasons.append(
                    f"backhand root left distance {result.backhand_root_left_distance:.3f} m "
                    f"> {self._max_backhand_root_left_distance_m:.3f} m")
            print(f"reject IK {attempt}/{self._resample_attempts}: " + "; ".join(reasons))
        print(f"warning: best IK after {self._resample_attempts} attempts — "
              f"axis {best_result.axis_err:.1f}°"
              + ("" if best_result.backhand_root_left_distance is None
                 else f", backhand root left distance {best_result.backhand_root_left_distance:.3f} m"))
        self._apply_ik_result(best_result)
        return best_result.target_qpos

    def _ik_attempt(
        self,
        seed_qpos: np.ndarray,
        seed_variant: int = 0,
    ) -> IKAttemptResult:
        """Solve IK once for the current command using one seed variant."""
        want_forehand = bool(self._command.is_forehand)
        hit_pos = self._command.target_hit_pos.astype(np.float64)
        hit_vel = self._command.target_hit_vel.astype(np.float64)
        target_axis = hit_vel / (np.linalg.norm(hit_vel) + 1e-8)
        if not want_forehand:
            target_axis = -target_axis

        offsets_s = self._target_frame_offsets / float(self.FPS)
        racket_targets = hit_pos[None, :] + offsets_s[:, None] * hit_vel[None, :]

        pred_tokens = self._fixed_pred_num_tokens()
        sampled_hit_time_s = self._hit_time_from_pred_tokens(pred_tokens)

        qpos_seed = seed_qpos.astype(np.float64).copy()
        qpos_seed[3:7] = RacketCenterIK._safe_quat(qpos_seed)
        qpos_seed[3:7] = np.array([np.cos(self._target_heading * 0.5), 0., 0.,
                                   np.sin(self._target_heading * 0.5)])

        hit_qpos = self._ik.solve(
            qpos_seed, hit_pos, target_axis, want_forehand,
            seed_variant=seed_variant)
        backhand_root_left_distance = None
        backhand_root_ok = True
        if not want_forehand and self._max_backhand_root_left_distance_m is not None:
            backhand_root_left_distance = max(0.0, float(hit_qpos[1] - hit_pos[1]))
            backhand_root_ok = (
                backhand_root_left_distance <= self._max_backhand_root_left_distance_m
            )

        qposes, racket_positions, axis_errors = [], [], []
        q_seed = hit_qpos.copy()
        for target, target_enabled in zip(racket_targets, self._target_pose_mask):
            if target_enabled:
                q = self._ik.solve(
                    q_seed, target, target_axis, want_forehand,
                    lock_root=True, apply_seed_posture=False)
                q_seed = q.copy()
            else:
                q = q_seed.copy()
            qposes.append(q.copy())

        mask_indices = np.flatnonzero(self._target_pose_mask).tolist()
        keyframe_idx = mask_indices[0]
        for q in qposes:
            racket_positions.append(self._ik.forward_site_pos(q))
            ax = self._ik.forward_site_y_axis(q)
            axis_errors.append(float(np.degrees(np.arccos(np.clip(ax @ target_axis, -1., 1.)))))

        axis_err = max(axis_errors[i] for i in mask_indices)
        ik_err = max(np.linalg.norm(racket_positions[i] - racket_targets[i]) for i in mask_indices)
        result = IKAttemptResult(
            target_qpos=np.stack(qposes).astype(np.float32),
            axis_err=axis_err,
            backhand_root_ok=backhand_root_ok,
            backhand_root_left_distance=backhand_root_left_distance,
            root_target=qposes[keyframe_idx][:3].astype(np.float32),
            hit_pos=hit_pos.astype(np.float32),
            hit_vel=hit_vel.astype(np.float32),
            racket_positions=np.stack(racket_positions).astype(np.float32),
            keyframe_qposes=np.stack(qposes).astype(np.float32),
            keyframe_qpos=qposes[keyframe_idx].astype(np.float32),
            pred_num_tokens=pred_tokens,
            mode_name="pingpong_forehand" if want_forehand else "pingpong_backhand",
        )
        print(
            f"sample {result.mode_name}: hit={np.round(hit_pos, 3)}, hit_vel={np.round(hit_vel, 3)}, "
            f"ik_err={ik_err:.4f} m, axis_err={axis_err:.1f}°, "
            f"tokens={pred_tokens}, hit_t={sampled_hit_time_s:.3f}s, ik_seed={seed_variant}"
            + (
                "" if backhand_root_left_distance is None
                else f", bh_root_left={backhand_root_left_distance:.3f}m"
            ))
        return result

    def _apply_ik_result(self, result: IKAttemptResult | None) -> None:
        if result is None:
            return
        self.latest_root_target = result.root_target
        self.latest_hit_pos = result.hit_pos
        self.latest_hit_vel = result.hit_vel
        self.latest_racket_positions = result.racket_positions
        self.latest_keyframe_qposes = result.keyframe_qposes
        self.latest_keyframe_qpos = result.keyframe_qpos
        self.latest_pred_num_tokens = result.pred_num_tokens
        self.latest_mode_name = result.mode_name


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def update_xml_command_markers(
        model: mujoco.MjModel, data: mujoco.MjData,
        hit_pos: np.ndarray | None, hit_vel: np.ndarray | None,
        scale: float = 0.25) -> None:
    if hit_pos is None or hit_vel is None or model.nmocap < 3:
        return
    hit_pos = np.asarray(hit_pos, dtype=np.float64)
    tip = hit_pos + np.asarray(hit_vel, dtype=np.float64) * scale
    data.mocap_pos[0] = hit_pos
    data.mocap_pos[1] = tip
    data.mocap_pos[2] = 0.5 * (hit_pos + tip)
    direction = tip - hit_pos
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        data.mocap_quat[2] = R.align_vectors(
            [direction / norm], [[0., 0., 1.]])[0].as_quat(scalar_first=True)


def draw_racket_positions(viewer, positions: np.ndarray | None) -> None:
    if positions is None:
        return
    for idx, pos in enumerate(positions):
        if viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.028, 0., 0.]),
            np.asarray(pos, dtype=np.float64),
            np.eye(3).reshape(-1),
            np.array([0.1, 1.0, 0.25, 0.35 + 0.14 * idx], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_ghost_g1_colored(
    viewer,
    mj_model,
    ghost_data,
    ghost_opt,
    ghost_pert,
    qpos: np.ndarray | None,
    rgba: np.ndarray,
) -> None:
    if qpos is None:
        return
    ghost_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, ghost_data)
    start = viewer.user_scn.ngeom
    mujoco.mjv_addGeoms(
        mj_model,
        ghost_data,
        ghost_opt,
        ghost_pert,
        mujoco.mjtCatBit.mjCAT_DYNAMIC,
        viewer.user_scn,
    )
    for geom_idx in range(start, viewer.user_scn.ngeom):
        viewer.user_scn.geoms[geom_idx].rgba[:] = rgba


def draw_ik_keyframe_ghosts(
    viewer,
    mj_model,
    ghost_data,
    ghost_opt,
    ghost_pert,
    qposes: np.ndarray | None,
) -> None:
    if qposes is None:
        return
    light = np.array([0.55, 0.80, 1.00, 0.16], dtype=np.float32)
    dark = np.array([0.05, 0.25, 1.00, 0.38], dtype=np.float32)
    num = max(1, len(qposes) - 1)
    for idx, qpos in enumerate(qposes):
        alpha = float(idx) / float(num)
        rgba = light * (1.0 - alpha) + dark * alpha
        draw_ghost_g1_colored(viewer, mj_model, ghost_data, ghost_opt, ghost_pert, qpos, rgba)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE = "idle"
    HIT = "hit"
    RECOVERY = "recovery"


def _last_n_frames(agent: PingPongMotionAgent, n: int = 4) -> t.Tensor:
    """Last n frames of the current buffer (for explicit context passing)."""
    total = agent.frames["mujoco_qpos"].shape[1]
    return agent.frames["mujoco_qpos"][:, max(0, total - n):]


# ---------------------------------------------------------------------------
# Shared setup builder
# ---------------------------------------------------------------------------

def _resolve_ready_qpos(args, mj_model) -> np.ndarray:
    """Resolve the 'ready stance' qpos used as agent.initial_qpos /
    controller.default_qpos. Priority:
      1. args.assets.default_qpos
      2. mj_model.key_qpos[0]   (XML <key> keyframe)
      3. zeros + identity quat  (last-resort)
    """
    cfg_qpos = getattr(getattr(args, "assets", None), "default_qpos", None)
    if cfg_qpos is not None:
        return np.asarray(cfg_qpos, dtype=np.float32).copy()
    if mj_model.nkey > 0:
        return np.asarray(mj_model.key_qpos[0], dtype=np.float32).copy()
    qpos = np.zeros(mj_model.nq, dtype=np.float32)
    qpos[3] = 1.0  # w=1 for identity quat
    return qpos


def _resolve_command_cfg(args) -> dict:
    """Build the cfg dict consumed by PingPongCommandState. Accepts both the
    training-side flat form (args.command.hit_*) and the deploy-side nested
    form (args.command.hit_target.*).
    """
    cmd = getattr(args, "command", None)
    if cmd is None:
        return {"mixed_use_base_relative_y_split": True}
    try:
        from omegaconf import OmegaConf
        cmd_dict = (OmegaConf.to_container(cmd, resolve=True)
                    if not isinstance(cmd, dict) else dict(cmd))
    except Exception:
        cmd_dict = dict(cmd) if isinstance(cmd, dict) else dict(vars(cmd))
    hit_target = cmd_dict.pop("hit_target", None)
    if isinstance(hit_target, dict):
        cmd_dict = {**hit_target, **cmd_dict}
    cmd_dict.setdefault("mixed_use_base_relative_y_split", True)
    return cmd_dict


def _load_models(args):
    """Load root + pose models from checkpoint (shared by Demo and deploy)."""
    args.return_model_configs = True
    args.return_dataloader = False
    args.EXP = args.model.planner

    models, confs = test(args)
    root_config = _resolve_training_config_for_ckpt(args.model.root_ckpt, args.model.root_config)
    pose_config = _resolve_training_config_for_ckpt(args.model.pose_ckpt, args.model.pose_config)
    if args.model.root_ckpt:
        if root_config is None:
            raise ValueError("model.root_config required (cannot be inferred from root_ckpt)")
        models["root"] = _load_model_from_training_config(root_config, args.model.root_ckpt, "root")
    else:
        models["root"].load_state_dict(
            t.load(confs["root"].ckpt_path, map_location="cpu")["state_dict"])
    if args.model.pose_ckpt:
        if pose_config is None:
            raise ValueError("model.pose_config required (cannot be inferred from pose_ckpt)")
        models["pose"] = _load_model_from_training_config(pose_config, args.model.pose_ckpt, "pose")
    else:
        models["pose"].load_state_dict(
            t.load(confs["pose"].ckpt_path, map_location="cpu")["state_dict"])
    return models


def build_motionbricks_setup(args) -> SimpleNamespace:
    """Build the shared pieces (inferencer, mj_model, agent, controller, …)
    from a cfg. Both Demo and the deploy runtime use this so changes to the
    construction signature stay aligned automatically.

    Returns a namespace with fields:
        inferencer, mj_model, mj_data, agent, controller,
        ready_qpos, default_pred_num_tokens.
    """
    models = _load_models(args)
    inferencer = motion_inference(models, models["pose"].args)

    mj_model, mj_data = build_mj_simulator(
        args.assets.humanoid_xml, inferencer.motion_rep.fps)
    ready_qpos = _resolve_ready_qpos(args, mj_model)
    mj_data.qpos[:] = ready_qpos

    agent = PingPongMotionAgent(
        inferencer,
        device="cuda",
        skeleton_xml=args.assets.skeleton_xml,
        initial_qpos=ready_qpos,
        target_root_realignment=args.runtime.target_root_realignment,
        source_root_realignment=args.runtime.source_root_realignment,
        force_canonicalization=args.runtime.force_canonicalization,
        filter_qpos=args.runtime.pre_filter_qpos,
        skip_ending_target_cond=args.runtime.skip_ending_target_cond,
    ).to("cuda")

    root_min = int(inferencer._root_model.args["min_tokens"])
    root_max = int(inferencer._root_model.args["max_tokens"])
    pose_min = int(inferencer._pose_model.args["min_tokens"])
    pose_max = int(inferencer._pose_model.args["max_tokens"])
    compatible_min = max(root_min, pose_min)
    compatible_max = min(root_max, pose_max)
    if compatible_min > compatible_max:
        raise ValueError(
            f"Root [{root_min},{root_max}] and pose [{pose_min},{pose_max}] token ranges do not overlap")
    default_tokens = args.model.pred_num_tokens or compatible_min
    if not (compatible_min <= default_tokens <= compatible_max):
        raise ValueError(
            f"pred_num_tokens={default_tokens} outside [{compatible_min},{compatible_max}]")
    print(f"Token ranges: root=[{root_min},{root_max}], pose=[{pose_min},{pose_max}], "
          f"compatible=[{compatible_min},{compatible_max}], default={default_tokens}")

    ik = args.ik
    controller = PingPongTargetController(
        mj_model,
        min_token=root_min, max_token=root_max,
        allowed_min_token=compatible_min, allowed_max_token=compatible_max,
        default_pred_num_tokens=default_tokens,
        command_cfg=_resolve_command_cfg(args),
        target_heading=args.controller.target_heading,
        default_qpos=ready_qpos,
        ik_iterations=ik.iterations,
        ik_damping=ik.damping,
        racket_site_name=ik.racket_site_name,
        target_frame_offsets=args.controller.target_frame_offsets,
        target_pose_mask=args.controller.target_pose_mask,
        constrain_target_root=bool(args.controller.constrain_target_root),
        orientation_weight=ik.orientation_weight,
        max_axis_error_deg=getattr(ik, "max_axis_error_deg", 35.0),
        resample_attempts=getattr(ik, "resample_attempts", 8),
        max_root_x=ik.max_root_x,
        max_root_yaw_delta=ik.max_root_yaw_delta,
        max_waist_yaw=ik.max_waist_yaw,
        max_backhand_root_left_distance_m=getattr(
            ik, "max_backhand_root_left_distance_m", 0.45),
    )

    return SimpleNamespace(
        inferencer=inferencer,
        mj_model=mj_model,
        mj_data=mj_data,
        agent=agent,
        controller=controller,
        ready_qpos=ready_qpos,
        default_pred_num_tokens=default_tokens,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

class Demo:
    def __init__(self, args):
        setup = build_motionbricks_setup(args)
        self.inferencer = setup.inferencer
        self.mj_model = setup.mj_model
        self.mj_data = setup.mj_data
        self.agent = setup.agent
        self.controller = setup.controller

        self.ghost_data = mujoco.MjData(self.mj_model)
        self.ghost_opt = mujoco.MjvOption()
        self.ghost_pert = mujoco.MjvPerturb()

        # controller_dt passed to generate_new_frames (controls idle regen frequency)
        self.controller_dt = 8 / self.controller.FPS * args.runtime.generate_dt

        self.phase = Phase.IDLE
        self.keyboard = SafeKeyboardState()
        self._prev_n = False


# ---------------------------------------------------------------------------
# Per-frame update
# ---------------------------------------------------------------------------

def _step(demo: Demo) -> None:
    """Advance one frame and handle state-machine transitions."""
    qpos = demo.agent.get_next_frame()
    demo.mj_data.qpos[:] = qpos

    n_down = demo.keyboard.is_n_pressed()
    n_edge = n_down and not demo._prev_n
    demo._prev_n = n_down

    if n_edge:
        print("=== HIT triggered ===")
        ctrl = demo.controller.sample_hit_control(demo.mj_data.qpos.copy())
        ctrl["context_mujoco_qpos"] = demo.agent.get_context_mujoco_qpos()
        force_generate_and_trim(demo.agent, ctrl, demo.controller_dt)
        demo.phase = Phase.HIT
        return

    if demo.phase == Phase.IDLE:
        return

    elif demo.phase == Phase.HIT:
        if demo.agent.is_done():
            print("=== RECOVERY triggered ===")
            root_target = demo.mj_data.qpos[:3].copy()
            # root_target[0] = max(root_target[0]-0.05, 0.0)  # step back a bit from the table
            ctrl = demo.controller.recovery_control(root_target)
            ctrl["context_mujoco_qpos"] = _last_n_frames(demo.agent)
            force_generate_and_trim(demo.agent, ctrl, demo.controller_dt)
            demo.phase = Phase.RECOVERY

    elif demo.phase == Phase.RECOVERY:
        if demo.agent.is_done():
            print("=== back to IDLE ===")
            demo.phase = Phase.IDLE


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_scene(demo: Demo, viewer, args) -> None:
    mujoco.mj_forward(demo.mj_model, demo.mj_data)
    viewer.user_scn.ngeom = 0
    update_xml_command_markers(
        demo.mj_model, demo.mj_data,
        demo.controller.latest_hit_pos, demo.controller.latest_hit_vel,
        args.visualization.velocity_marker_scale)
    mujoco.mj_forward(demo.mj_model, demo.mj_data)
    draw_pingpong_table(viewer, args.visualization.table_position, args.visualization.table_yaw)
    draw_root_target(viewer, demo.controller.latest_root_target)
    draw_model_target_roots(viewer, demo.agent.latest_model_target_roots)
    draw_racket_positions(viewer, demo.controller.latest_racket_positions)
    draw_ik_keyframe_ghosts(
        viewer, demo.mj_model, demo.ghost_data, demo.ghost_opt, demo.ghost_pert,
        demo.controller.latest_keyframe_qposes)
    viewer.cam.lookat[:] = demo.mj_data.qpos[:3]
    viewer.sync()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    demo = Demo(args)
    with mujoco.viewer.launch_passive(demo.mj_model, demo.mj_data) as viewer:
        disable_mujoco_keyboard_shortcuts("n")
        for _ in range(args.runtime.max_steps):
            step_start = time.time()
            _step(demo)
            draw_scene(demo, viewer, args)
            elapsed = time.time() - step_start
            sleep = demo.mj_model.opt.timestep - elapsed
            if sleep > 0:
                time.sleep(sleep)
            if not viewer.is_running():
                break


def parse_args(argv: list[str] | None = None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(root, "configs/pingpong_g1.yaml")
    parser = argparse.ArgumentParser(
        description=(
            f"{__doc__}\n\n"
            "Most options live in configs/pingpong_g1.yaml. "
            "Override with dot-list syntax, e.g. ik.max_root_x=0.12"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=default_config, help="YAML config file.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides.")
    cli = parser.parse_args(argv)

    cfg = OmegaConf.load(cli.config)
    if cli.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli.overrides))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    def repo_path(p):
        return p if p is None or os.path.isabs(p) else os.path.join(root, p)

    assets, model = cfg["assets"], cfg["model"]
    for key in ("humanoid_xml", "skeleton_xml",
                "result_dir", "data_root", "explicit_dataset_folder"):
        assets[key] = repo_path(assets[key])
    for key in ("root_ckpt", "root_config", "pose_ckpt", "pose_config"):
        model[key] = repo_path(model[key])

    return SimpleNamespace(
        # flat fields required by motionbricks.exp_setup.experiment.test()
        result_dir=assets["result_dir"],
        data_root=assets["data_root"],
        explicit_dataset_folder=assets["explicit_dataset_folder"],
        # structured sections
        assets=SimpleNamespace(**assets),
        model=SimpleNamespace(**model),
        runtime=SimpleNamespace(**cfg["runtime"]),
        controller=SimpleNamespace(**cfg["controller"]),
        ik=SimpleNamespace(**cfg["ik"]),
        command=SimpleNamespace(**cfg["command"]),
        visualization=SimpleNamespace(**cfg["visualization"]),
    )


if __name__ == "__main__":
    main(parse_args())
