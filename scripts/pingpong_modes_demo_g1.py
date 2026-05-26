#!/usr/bin/env python3
"""Keyboard mode demo for three ping-pong clips.

Hold F for forehand, hold B for backhand, release keys to return to frozen.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np
import torch as t
from scipy.spatial.transform import Rotation as R

from motionbricks.exp_setup.experiment import test
from motionbricks.motion_backbone.demo import clips as demo_clips
from motionbricks.motion_backbone.demo.full_agent import angle_to_Z_rotation_matrix, full_navigation_agent
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.helper.mujoco_helper import get_mujoco_converter


PINGPONG_CLIPS = {
    "frozen_default": {
        "allowed_pred_num_tokens": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "avg_root_vel": 0.0,
    },
    "pingpong_forehand": {
        "allowed_pred_num_tokens": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "avg_root_vel": 0.0,
    },
    "pingpong_backhand": {
        "allowed_pred_num_tokens": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "avg_root_vel": 0.0,
    },
}


class PingPongClipHolder(demo_clips.clip_holder):
    CLIPS = PINGPONG_CLIPS
    DEFAULT_KEYS = {
        "frozen_default": "",
        "pingpong_forehand": "f",
        "pingpong_backhand": "b",
    }

    def _apply_root_headings_correction(self):
        pass

    def blendspace_modes_remap_from_velocity(self, mode: t.Tensor,
                                             target_movement_direction: t.Tensor,
                                             target_heading: t.Tensor):
        return mode


class PingPongNavigationAgent(full_navigation_agent):
    def __init__(
        self,
        inferencer: motion_inference,
        device: str = "cuda",
        speed_scale: list[float] = [1.0, 1.0],
        target_root_realignment: bool = True,
        source_root_realignment: bool = True,
        pred_offsets: int = full_navigation_agent.DEFAULT_PRED_OFFSETS,
        skeleton_xml: str = "assets/skeletons/g1/g1.xml",
        filter_qpos: bool = True,
        force_canonicalization: bool = True,
        bypass_spring_model: bool = True,
        skip_ending_target_cond: bool = True,
        ckpt_path: str | None = None,
    ):
        t.nn.Module.__init__(self)
        self._inferencer = inferencer.eval().to(device)
        self._motion_rep = deepcopy(inferencer.motion_rep).to(device)
        self._converter = get_mujoco_converter(self._motion_rep, skeleton_xml).to(device)
        self._clip_holder = PingPongClipHolder(
            ckpt_path=ckpt_path,
            converter=self._converter,
        )
        self._train_dataloader = None
        self._device = device
        self._fps = self._motion_rep.fps
        self._target_root_realignment = target_root_realignment
        self._source_root_realignment = source_root_realignment

        self.PRED_OFFSETS = pred_offsets
        self.FILTER_QPOS = filter_qpos
        self.FORCE_CANONICALIZATION = force_canonicalization
        self.BYPASS_SPRING_MODEL = bypass_spring_model
        self.SKIP_ENDING_TARGET_COND = skip_ending_target_cond

        self.frames = {
            "model_features": None,
            "mujoco_qpos": None,
            "mode": None,
        }
        self._speed_scale_min, self._speed_scale_max = speed_scale
        self._initialize_frames()
        self._has_prebaked_inference_engine = False
        self.latest_model_target_roots = None
        self._active_target_window_end_frame = None

    def _initialize_frames(self):
        self._current_frame_idx = 0
        self.frames["model_features"] = self._clip_holder.motion_feature[0, :self._clip_holder.num_frames_per_clip[0]][None]
        self.frames["mujoco_qpos"] = self._clip_holder.mujoco_qpos[0, :self._clip_holder.num_frames_per_clip[0]][None]
        num_min_frames_in_buffer = 64
        if self.frames["mujoco_qpos"].shape[1] < num_min_frames_in_buffer:
            repeat_count = num_min_frames_in_buffer - self.frames["mujoco_qpos"].shape[1]
            self.frames["mujoco_qpos"] = t.cat(
                [self.frames["mujoco_qpos"], self.frames["mujoco_qpos"][:, -1:].repeat(1, repeat_count, 1)],
                dim=1,
            )
            self.frames["model_features"] = t.cat(
                [self.frames["model_features"], self.frames["model_features"][:, -1:].repeat(1, repeat_count, 1)],
                dim=1,
            )

    def _should_regenerate(self, input: dict):
        frozen_mode_id = list(self._clip_holder.CLIPS.keys()).index("frozen_default")
        if self.frames["mode"] is not None and self.frames["mode"].item() == frozen_mode_id and \
                input["mode"].item() == frozen_mode_id:
            return False
        return True

    def should_defer_regeneration_until_target_window_end(self, force_generation: bool = False) -> bool:
        if force_generation or self._active_target_window_end_frame is None:
            return False
        end_frame = min(self._active_target_window_end_frame, self.frames["mujoco_qpos"].shape[1] - 1)
        if self._current_frame_idx < end_frame:
            return True
        self._active_target_window_end_frame = None
        return False

    def generate_new_frames(self, input: dict, controller_dt: float = 0.25, force_generation: bool = False):
        result = super().generate_new_frames(input, controller_dt, force_generation)
        if force_generation:
            num_pred_frames = self.frames.get("num_pred_frames", None)
            if num_pred_frames is not None:
                # Target pose conditions occupy the final four generated frames.
                self._active_target_window_end_frame = max(0, int(num_pred_frames.item()) - 1)
        return result

    def _generate_target_joint_transforms(self, input: dict):
        output = super()._generate_target_joint_transforms(input)
        roots_motion = output[2].detach()
        roots_mujoco = t.matmul(
            self._converter.motion_to_mujoco_matrix.to(roots_motion.device, roots_motion.dtype)[None, None],
            roots_motion[..., None],
        )[..., 0]
        if self.FORCE_CANONICALIZATION and "first_frame_heading_angle" in input:
            rot = angle_to_Z_rotation_matrix(input["first_frame_heading_angle"]).to(roots_mujoco.device, roots_mujoco.dtype)
            roots_mujoco = t.matmul(rot[:, None], roots_mujoco[..., None])[..., 0]
            first_pos = input["first_frame_position"].to(roots_mujoco.device, roots_mujoco.dtype)
            roots_mujoco = roots_mujoco - roots_mujoco[:, :1] * t.tensor(
                [[[1.0, 1.0, 0.0]]], device=roots_mujoco.device, dtype=roots_mujoco.dtype
            ) + first_pos[:, None]
        self.latest_model_target_roots = roots_mujoco[0].detach().cpu().numpy()
        return output


class KeyboardState:
    def __init__(self):
        self._pressed = set()
        if platform.system() in ("Linux", "Darwin"):
            from pynput import keyboard

            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        else:
            self._keyboard = __import__("keyboard")

    def _on_press(self, key):
        char = getattr(key, "char", None) or getattr(key, "name", None)
        if char:
            self._pressed.add(char.lower())

    def _on_release(self, key):
        char = getattr(key, "char", None) or getattr(key, "name", None)
        if char:
            self._pressed.discard(char.lower())

    def snapshot(self) -> dict[str, bool]:
        if platform.system() in ("Linux", "Darwin"):
            return {"f": "f" in self._pressed, "b": "b" in self._pressed}
        return {"f": self._keyboard.is_pressed("f"), "b": self._keyboard.is_pressed("b")}


class PingPongModeController:
    def __init__(
        self,
        min_token: int,
        max_token: int,
        fixed_frames: dict[str, int | None] | None = None,
        root_lateral_range: tuple[float, float] = (-0.76, 0.76),
        target_heading: float = 0.0,
    ):
        self._FPS = 30
        self._CONTROLLER_DT = 8 / self._FPS
        self._NUM_HISTORY_STEPS = 5
        self._prev_qpos = None
        self._keys = KeyboardState()
        self._min_token = min_token
        self._max_token = max_token
        self._fixed_frames = fixed_frames or {}
        self._root_lateral_range = root_lateral_range
        self._target_heading = target_heading
        self._prev_keys = {"f": False, "b": False}
        self.latest_root_target = None
        self.latest_root_heading = 0.0
        self.latest_mode_name = "frozen_default"
        self.latest_frame = 0

    def get_controller_dt(self):
        return self._CONTROLLER_DT

    def get_prev_qpos(self):
        return self._prev_qpos.copy()

    def _allowed_pred_num_tokens(self, mode_name: str):
        allowed = PINGPONG_CLIPS[mode_name].get("allowed_pred_num_tokens")
        if allowed is not None:
            return t.tensor(allowed, dtype=t.int).view(1, -1)
        return t.ones(self._max_token - self._min_token + 1, dtype=t.int).view(1, -1)

    @staticmethod
    def _safe_root_quat(qpos: np.ndarray) -> np.ndarray:
        quat = np.asarray(qpos[3:7], dtype=np.float64).copy()
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    @staticmethod
    def _root_heading_from_qpos(qpos: np.ndarray) -> float:
        root_quat = PingPongModeController._safe_root_quat(qpos)
        facing = R.from_quat(root_quat, scalar_first=True).apply(np.array([1.0, 0.0, 0.0]))
        return float(np.arctan2(facing[1], facing[0]))

    def _sample_specific_root_target(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root = qpos[:3].astype(np.float32)
        current_heading = self._root_heading_from_qpos(qpos)
        left = np.array([-np.sin(current_heading), np.cos(current_heading), 0.0], dtype=np.float32)
        lateral = np.random.uniform(*self._root_lateral_range)
        target_root = root + left * lateral
        self.latest_root_target = target_root.copy()
        self.latest_root_heading = self._target_heading

        positions = np.repeat(target_root[None, :], 4, axis=0).astype(np.float32)
        headings = np.repeat(np.array([self._target_heading], dtype=np.float32), 4, axis=0)
        return positions, headings

    def generate_control_signals(self, viewer, mj_model: mujoco.MjModel, mj_data: mujoco.MjData,
                                 visualize: bool = True, control_info: dict | None = None):
        if self._prev_qpos is None:
            self._prev_qpos = np.zeros((self._NUM_HISTORY_STEPS, mj_model.nq))
            initial_qpos = mj_data.qpos.copy()
            initial_qpos[3:7] = self._safe_root_quat(initial_qpos)
            self._prev_qpos[:] = initial_qpos.reshape(1, -1)

        keys = self._keys.snapshot()
        forehand_edge = keys["f"] and not self._prev_keys["f"]
        backhand_edge = keys["b"] and not self._prev_keys["b"]
        self._prev_keys = keys

        if forehand_edge:
            mode_name = "pingpong_forehand"
        elif backhand_edge:
            mode_name = "pingpong_backhand"
        else:
            mode_name = "frozen_default"

        mode = t.tensor([[list(PINGPONG_CLIPS.keys()).index(mode_name)]], dtype=t.long)
        root_quat = self._safe_root_quat(self._prev_qpos[-1])
        facing = R.from_quat(root_quat, scalar_first=True).apply(np.array([1.0, 0.0, 0.0]))
        facing = facing * np.array([1.0, 1.0, 0.0])
        if np.linalg.norm(facing) < 1e-4:
            facing = np.array([1.0, 0.0, 0.0])
        facing = (facing / (np.linalg.norm(facing) + 1e-5)).astype(np.float32)

        current_qpos = mj_data.qpos.copy()
        current_qpos[3:7] = self._safe_root_quat(current_qpos)
        self._prev_qpos = np.concatenate((self._prev_qpos[1:], current_qpos.reshape(1, -1)), axis=0)
        control = {
            "movement_direction": t.zeros(1, 3),
            "facing_direction": t.tensor([[np.cos(self._target_heading), np.sin(self._target_heading), 0.0]]).float(),
            "mode": mode,
            "allowed_pred_num_tokens": self._allowed_pred_num_tokens(mode_name),
        }
        fixed_frame = self._fixed_frames.get(mode_name)
        if fixed_frame is not None:
            fixed_frame = int(fixed_frame)
            control["random_seed"] = t.tensor([fixed_frame], dtype=t.int)
        if mode_name != "frozen_default":
            self.latest_mode_name = mode_name
            if fixed_frame is not None:
                self.latest_frame = fixed_frame
            root_positions, root_headings = self._sample_specific_root_target(current_qpos)
            control.update({
                "specific_target_positions": t.from_numpy(root_positions).view(1, 4, 3).float(),
                "specific_target_headings": t.from_numpy(root_headings).view(1, 4).float(),
                "has_specific_target": t.ones(1, 1, dtype=t.int),
                "force_generation": True,
            })
        return control


def build_mj_simulator(humanoid_xml: str, fps: int):
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 1 / fps
    return mj_model, mj_data


def disable_mujoco_keyboard_shortcuts(controller_keys: str = "fb") -> None:
    if platform.system() != "Linux":
        return
    try:
        from Xlib import X
        from Xlib import display as xdisplay

        xdpy = xdisplay.Display()
        root = xdpy.screen().root

        def find_window_by_name(win, name_substr):
            try:
                name = win.get_wm_name()
                if name and name_substr in name:
                    return win
            except Exception:
                pass
            for child in win.query_tree().children:
                result = find_window_by_name(child, name_substr)
                if result:
                    return result
            return None

        time.sleep(0.5)
        mj_win = find_window_by_name(root, "MuJoCo")
        if mj_win:
            for ch in controller_keys:
                keycode = xdpy.keysym_to_keycode(ord(ch.upper()))
                mj_win.grab_key(keycode, X.AnyModifier, False, X.GrabModeAsync, X.GrabModeAsync)
            xdpy.sync()
    except Exception as exc:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {exc}")


def draw_pingpong_table(viewer, position: tuple[float, float, float], yaw: float) -> None:
    if viewer.user_scn.ngeom + 2 >= viewer.user_scn.maxgeom:
        return
    center = np.array(position, dtype=np.float64)
    mat = R.from_euler("z", yaw).as_matrix().astype(np.float64).reshape(-1)

    table_idx = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[table_idx],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([1.37, 0.76, 0.025], dtype=np.float64),
        center,
        mat,
        np.array([0.05, 0.22, 0.16, 0.55], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1

    net_center = center.copy()
    net_center[2] = 0.89
    net_idx = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[net_idx],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([0.015, 0.76, 0.08], dtype=np.float64),
        net_center,
        mat,
        np.array([0.92, 0.92, 0.88, 0.75], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_root_target(viewer, target: np.ndarray | None) -> None:
    if target is None or viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
        return
    pos = np.asarray(target, dtype=np.float64).copy()
    pos[2] = max(pos[2], 0.08)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.08, 0.0, 0.0], dtype=np.float64),
        pos,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.array([1.0, 0.15, 0.08, 0.85], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_model_target_roots(viewer, targets: np.ndarray | None) -> None:
    if targets is None:
        return
    for target in targets:
        if viewer.user_scn.ngeom + 1 >= viewer.user_scn.maxgeom:
            return
        pos = np.asarray(target, dtype=np.float64).copy()
        pos[2] = max(pos[2], 0.08)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.045, 0.0, 0.0], dtype=np.float64),
            pos,
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array([1.0, 0.85, 0.05, 0.9], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def _heading_to_wxyz_quat(heading: float) -> np.ndarray:
    return np.array([np.cos(heading * 0.5), 0.0, 0.0, np.sin(heading * 0.5)], dtype=np.float64)


def build_ghost_keyframe_qpos(agent: PingPongNavigationAgent, controller: PingPongModeController) -> np.ndarray | None:
    if controller.latest_root_target is None:
        return None
    mode_idx = list(PINGPONG_CLIPS.keys()).index(controller.latest_mode_name)
    num_frames = int(agent._clip_holder.num_frames_per_clip[mode_idx].item())
    frame = max(0, min(controller.latest_frame, num_frames - 1))
    qpos = agent._clip_holder.mujoco_qpos[mode_idx, frame].detach().cpu().numpy().copy()
    qpos[:3] = controller.latest_root_target
    clip_heading = float(agent._clip_holder.global_headings[mode_idx, frame].detach().cpu().item())
    diff_heading = (controller.latest_root_heading - clip_heading + np.pi) % (2.0 * np.pi) - np.pi
    root_rot = R.from_quat(qpos[3:7], scalar_first=True)
    qpos[3:7] = (R.from_euler("z", diff_heading) * root_rot).as_quat(scalar_first=True)
    return qpos


def draw_ghost_g1(viewer, mj_model, ghost_data, ghost_opt, ghost_pert, qpos: np.ndarray | None) -> None:
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
        viewer.user_scn.geoms[geom_idx].rgba[:] = np.array([0.2, 0.55, 1.0, 0.28], dtype=np.float32)


class Demo:
    def __init__(self, args):
        args.return_model_configs = True
        args.return_dataloader = False
        args.EXP = args.planner

        models, confs = test(args)
        for name in ("pose", "root"):
            models[name].load_state_dict(t.load(confs[name].ckpt_path)["state_dict"])
        self.inferencer = motion_inference(models, models["pose"].args)
        self.full_agent = PingPongNavigationAgent(
            self.inferencer,
            device="cuda",
            skeleton_xml=args.skeleton_xml,
            ckpt_path=args.clips_ckpt,
            target_root_realignment=args.target_root_realignment,
            source_root_realignment=args.source_root_realignment,
            force_canonicalization=args.force_canonicalization,
            filter_qpos=args.pre_filter_qpos,
            skip_ending_target_cond=args.skip_ending_target_cond,
        ).to("cuda")
        self.controller = PingPongModeController(
            min_token=self.inferencer._args["min_tokens"],
            max_token=self.inferencer._args["max_tokens"],
            fixed_frames={
                "frozen_default": args.frozen_frame,
                "pingpong_forehand": args.forehand_frame,
                "pingpong_backhand": args.backhand_frame,
            },
            root_lateral_range=args.root_lateral_range,
            target_heading=args.target_heading,
        )
        self.mj_model, self.mj_data = build_mj_simulator(args.humanoid_xml, self.inferencer.motion_rep.fps)
        self.ghost_data = mujoco.MjData(self.mj_model)
        self.ghost_opt = mujoco.MjvOption()
        self.ghost_pert = mujoco.MjvPerturb()


def main(args):
    demo = Demo(args)
    steps = 0
    with mujoco.viewer.launch_passive(demo.mj_model, demo.mj_data) as viewer:
        disable_mujoco_keyboard_shortcuts()
        while viewer.is_running() and steps < args.max_steps:
            steps += 1
            step_start = time.time()
            qpos = demo.full_agent.get_next_frame()
            context_qpos = demo.full_agent.get_context_mujoco_qpos()
            demo.mj_data.qpos[:] = qpos

            control = demo.controller.generate_control_signals(viewer, demo.mj_model, demo.mj_data)
            control["context_mujoco_qpos"] = context_qpos
            force_generation = bool(control.pop("force_generation", False))
            skip_generation = bool(control.pop("skip_generation", False))
            if demo.full_agent.should_defer_regeneration_until_target_window_end(force_generation):
                skip_generation = True
            if not skip_generation:
                with t.no_grad():
                    demo.full_agent.generate_new_frames(
                        control,
                        demo.controller.get_controller_dt() * args.generate_dt,
                        force_generation=force_generation,
                    )

            mujoco.mj_forward(demo.mj_model, demo.mj_data)
            viewer.user_scn.ngeom = 0
            draw_pingpong_table(viewer, args.table_position, args.table_yaw)
            draw_root_target(viewer, demo.controller.latest_root_target)
            draw_model_target_roots(viewer, demo.full_agent.latest_model_target_roots)
            draw_ghost_g1(
                viewer,
                demo.mj_model,
                demo.ghost_data,
                demo.ghost_opt,
                demo.ghost_pert,
                build_ghost_keyframe_qpos(demo.full_agent, demo.controller),
            )
            viewer.cam.lookat[:] = demo.controller.get_prev_qpos()[:, :3].mean(axis=0)
            viewer.sync()
            sleep_time = demo.mj_model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def parse_args():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def float_pair(text: str) -> tuple[float, float]:
        values = tuple(float(v) for v in text.split(","))
        if len(values) != 2:
            raise argparse.ArgumentTypeError(f"expected MIN,MAX, got {text!r}")
        return values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanoid_xml", default=os.path.join(root, "assets/skeletons/g1/scene_29dof.xml"))
    parser.add_argument("--skeleton_xml", default=os.path.join(root, "assets/skeletons/g1/g1.xml"))
    parser.add_argument("--clips_ckpt", default=os.path.join(root, "out/G1-pingpong-modes-clip.ckpt"))
    parser.add_argument("--result_dir", default=os.path.join(root, "out"))
    parser.add_argument("--data_root", default=os.path.join(root, "datasets"))
    parser.add_argument("--explicit_dataset_folder", default=os.path.join(root, "datasets/motionbricks-G1"))
    parser.add_argument("--planner", default="default")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--frozen_frame", type=int, default=0)
    parser.add_argument("--forehand_frame", type=int, default=0)
    parser.add_argument("--backhand_frame", type=int, default=0)
    parser.add_argument("--root_lateral_range", type=float_pair, default=(-0.35, 0.35), metavar="MIN,MAX")
    parser.add_argument("--target_heading", type=float, default=0.0)
    parser.add_argument("--table_position", type=lambda s: tuple(float(v) for v in s.split(",")),
                        default=(1.87, 0.0, 0.74), metavar="X,Y,Z")
    parser.add_argument("--table_yaw", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
