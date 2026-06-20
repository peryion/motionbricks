"""Ping-pong hit command sampling + observation-vector building.

`PingPongCommandState` owns the random hit target (hit_pos, hit_vel, root_xy)
sampled on each `resample()` from cfg-defined ranges, plus the time-based state
(`elapsed_s`, `time_to_hit_frozen`) needed to build the policy command vector
during demos and buffer generation.
"""

import numpy as np


class PingPongCommandState:
    def __init__(self, cfg, *, zero_root_target: bool = False):
        self.cfg = cfg
        self.zero_root_target = zero_root_target
        self.elapsed_s = 0.0
        self.hit_time_s = 0.6
        self.next_resample_time_s = 0.7
        self.command_id = 0
        self.is_forehand = False
        self.time_to_hit_frozen = False
        self.target_hit_pos = np.zeros(3, dtype=np.float32)
        self.target_hit_vel = np.zeros(3, dtype=np.float32)
        self.target_root_xy = np.zeros(2, dtype=np.float32)
        self.resample(base_y=0.0)

    def _sample_uniform(self, lo, hi):
        return float(np.random.uniform(lo, hi))

    def resample(self, base_y: float = 0.0):
        self.elapsed_s = 0.0
        self.command_id += 1
        self.time_to_hit_frozen = False
        fixed_next = self.cfg.get("fixed_next_resample_time_s", None)
        if fixed_next is None:
            long_pause_prob = float(self.cfg.get("long_pause_prob", 0.0))
            if np.random.rand() < min(max(long_pause_prob, 0.0), 1.0):
                lo, hi = self.cfg.get("long_next_resample_time_range_s", [3.0, 6.0])
            else:
                lo, hi = self.cfg.get("next_resample_time_range_s", [0.7, 2.5])
            self.next_resample_time_s = self._sample_uniform(lo, hi)
        else:
            self.next_resample_time_s = float(fixed_next)

        hit_plane_x = float(self.cfg.get("hit_plane_x", 0.4))
        z_lo, z_hi = self.cfg.get("hit_z_range", [0.76, 1.26])
        vx_lo, vx_hi = self.cfg.get("hit_vel_x_range", [1.3, 2.0])
        vy_lo, vy_hi = self.cfg.get("hit_vel_y_range", [-0.35, 0.35])
        vz_lo, vz_hi = self.cfg.get("hit_vel_z_range", [0.0, 0.4])
        root_target_x = float(self.cfg.get("root_target_x", 0.0))

        if bool(self.cfg.get("mixed_use_base_relative_y_split", True)):
            fore_range = self.cfg.get("forehand_hit_y_range", [-0.7, 0.7])
            back_range = self.cfg.get("backhand_hit_y_range", [-0.7, 0.7])
            fore_low, fore_high = float(fore_range[0]), float(fore_range[1])
            back_low, back_high = float(back_range[0]), float(back_range[1])
            split_y = float(base_y) + float(self.cfg.get("mixed_backhand_split_offset_y", -0.2))
            fore_interval = (fore_low, min(fore_high, split_y))
            back_interval = (max(back_low, split_y), back_high)
            valid = []
            if fore_interval[1] >= fore_interval[0]:
                valid.append((True, fore_interval))
            if back_interval[1] >= back_interval[0]:
                valid.append((False, back_interval))
            if not valid:
                self.is_forehand = bool(np.random.rand() < 0.5)
                low, high = (fore_low, fore_high) if self.is_forehand else (back_low, back_high)
            else:
                self.is_forehand, (low, high) = valid[int(np.random.randint(len(valid)))]
        else:
            low, high = self.cfg.get("hit_y_range", [-0.7, 0.7])
            low, high = float(low), float(high)
        if high < low:
            high = low
        y = self._sample_uniform(low, high)
        z = self._sample_uniform(z_lo, z_hi)
        self.target_hit_pos[:] = np.array([hit_plane_x, y, z], dtype=np.float32)
        self.target_hit_vel[:] = np.array(
            [
                self._sample_uniform(vx_lo, vx_hi),
                self._sample_uniform(vy_lo, vy_hi),
                self._sample_uniform(vz_lo, vz_hi),
            ],
            dtype=np.float32,
        )
        self.target_root_xy[:] = np.array([root_target_x, y], dtype=np.float32)

    def update(self, dt):
        self.elapsed_s += dt
        if bool(self.cfg.get("freeze_to_ready_after_hit", False)):
            freeze_delay_s = float(self.cfg.get("freeze_to_ready_delay_after_hit_s", 0.0))
            if self.elapsed_s >= self.hit_time_s + freeze_delay_s:
                self.time_to_hit_frozen = True

    def _mid_command_vec(self, time_to_hit: float) -> np.ndarray:
        mid_hit_y = 0.5 * (
            float(self.cfg.get("forehand_hit_y_range", [-0.7, 0.7])[0])
            + float(self.cfg.get("backhand_hit_y_range", [-0.7, 0.7])[1])
        )
        hit_z_range = self.cfg.get("hit_z_range", [0.76, 1.36])
        hit_vel_x_range = self.cfg.get("hit_vel_x_range", [1.3, 2.0])
        hit_vel_y_range = self.cfg.get("hit_vel_y_range", [-0.35, 0.35])
        hit_vel_z_range = self.cfg.get("hit_vel_z_range", [0.0, 0.4])
        mid_hit_z = 0.5 * (float(hit_z_range[0]) + float(hit_z_range[1]))
        mid_vx = 0.5 * (float(hit_vel_x_range[0]) + float(hit_vel_x_range[1]))
        mid_vy = 0.5 * (float(hit_vel_y_range[0]) + float(hit_vel_y_range[1]))
        mid_vz = 0.5 * (float(hit_vel_z_range[0]) + float(hit_vel_z_range[1]))
        hit_plane_x = float(self.cfg.get("hit_plane_x", 0.4))
        return np.array(
            [time_to_hit, hit_plane_x, mid_hit_y, mid_hit_z, mid_vx, mid_vy, mid_vz, 0.0, 0.0],
            dtype=np.float32,
        )

    def command_vec(self):
        time_to_hit = self.hit_time_s - self.elapsed_s
        min_time_to_hit = self.cfg.get("min_time_to_hit_s", -0.9)
        if min_time_to_hit is not None:
            time_to_hit = max(time_to_hit, float(min_time_to_hit))
        if bool(self.cfg.get("neutralize_command_when_frozen", False)) and self.time_to_hit_frozen:
            return self._mid_command_vec(time_to_hit)
        root_target = np.zeros(2, dtype=np.float32) if self.zero_root_target else self.target_root_xy.astype(np.float32)
        return np.concatenate(
            [
                np.array([time_to_hit], dtype=np.float32),
                self.target_hit_pos.astype(np.float32),
                self.target_hit_vel.astype(np.float32),
                root_target,
            ],
            axis=0,
        )

    def stroke_sign(self):
        min_time_to_hit = self.cfg.get("min_time_to_hit_s", -0.9)
        min_time_reached = (
            min_time_to_hit is not None
            and self.hit_time_s - self.elapsed_s <= float(min_time_to_hit) + 1e-6
        )
        if self.time_to_hit_frozen or min_time_reached:
            return np.array([0.0], dtype=np.float32)
        return np.array([1.0 if self.is_forehand else -1.0], dtype=np.float32)

    def stroke_name(self):
        return "forehand" if self.is_forehand else "backhand"
