"""Shared utilities for the ping-pong MotionBricks demos."""

from __future__ import annotations

from copy import deepcopy
import os
import platform
import select
import sys
import termios
import time
import tty

from hydra.utils import instantiate
import mujoco
import numpy as np
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation as R
import torch as t

from motionbricks.helper.pl_util import load_motion_rep


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

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([1.37, 0.76, 0.025], dtype=np.float64),
        center,
        mat,
        np.array([0.05, 0.22, 0.16, 0.55], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1

    net_center = center.copy()
    net_center[2] = 0.89
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
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


def _resolve_training_config_for_ckpt(ckpt_path: str | None, explicit_config: str | None) -> str | None:
    if explicit_config:
        return explicit_config
    if not ckpt_path:
        return None
    ckpt = os.path.abspath(ckpt_path)
    candidate = os.path.abspath(os.path.join(os.path.dirname(ckpt), "..", "config.yaml"))
    return candidate if os.path.exists(candidate) else None


def _load_model_from_training_config(config_path: str, ckpt_path: str, model_name: str):
    conf = OmegaConf.load(config_path)
    if "model" in conf and "train" in conf:
        conf = conf.model

    motion_rep = load_motion_rep(conf)
    if conf.model.pose_vqvae_network is not None:
        pose_vqvae_motion_rep = getattr(conf.model, "pose_vqvae_motion_rep", "local")
        pose_vqvae_motion_rep = (
            motion_rep.dual_rep.local_motion_rep
            if pose_vqvae_motion_rep == "local"
            else motion_rep.dual_rep.global_motion_rep
        )
        pose_vqvae_network = instantiate(conf.model.pose_vqvae_network, motion_rep=pose_vqvae_motion_rep)
    else:
        pose_vqvae_network = None

    model_conf = deepcopy(conf.model)
    for key in ("optimizer", "scheduler"):
        if key in model_conf:
            with open_dict(model_conf):
                del model_conf[key]
    backbone_network = instantiate(model_conf.backbone_network, motion_rep=motion_rep)
    model = instantiate(
        model_conf,
        pose_vqvae_network=pose_vqvae_network,
        backbone_network=backbone_network,
        motion_rep=motion_rep,
    )
    state = t.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    print(f"Loaded custom {model_name}: {ckpt_path}")
    return model


class TerminalKeyPoller:
    """Non-blocking stdin reader used by demo / deploy scripts to poll single
    keypresses without curses. cbreak mode is restored on close().
    """

    def __init__(self):
        self._enabled = False
        self._fd = None
        self._old = None
        if sys.stdin.isatty():
            try:
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._enabled = True
            except Exception:
                self._enabled = False

    @property
    def enabled(self):
        return self._enabled

    def poll_chars(self):
        if not self._enabled:
            return []
        chars = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            c = sys.stdin.read(1)
            if not c:
                break
            chars.append(c)
        return chars

    def close(self):
        if self._enabled and self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._enabled = False
