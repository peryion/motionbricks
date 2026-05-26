"""Standalone ping-pong demo agent.

Owns its own setup pipeline so it does not depend on navigation_demo.
The only controller it supports is PingPongTargetController.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch as t

from motionbricks.exp_setup.experiment import test
from motionbricks.motion_backbone.demo.controllers import PingPongTargetController
from motionbricks.motion_backbone.demo.utils import build_mj_simulator
from motionbricks.motion_backbone.inference.motion_inference import motion_inference


def _load_custom_bank(json_path: str | None) -> list[dict] | None:
    """Load a custom bank from a JSON file. Returns None if no path given."""
    if not json_path:
        return None
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"--pingpong_bank_json not found: {path}")
    with path.open() as f:
        bank = json.load(f)
    if not isinstance(bank, list) or len(bank) == 0:
        raise ValueError(f"Bank JSON must be a non-empty list of entries: {path}")
    for i, entry in enumerate(bank):
        entry.setdefault("id", i)
        entry.setdefault("name", f"entry_{i}")
    return bank


class PingPongDemo:
    """Self-contained demo agent wired to PingPongTargetController.

    Parameters mirror the CLI args in pingpong_demo_g1.py so the script can
    pass its parsed Namespace directly.
    """

    def __init__(self, args):
        self.args = args
        self._resolve_paths()
        self._load_models()
        self._build_controller()
        self._build_simulator()

    # ------------------------------------------------------------------
    # Setup steps
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """Fill in default file-system paths when not explicitly provided."""
        # utils.py lives at motionbricks/motion_backbone/demo/utils.py,
        # so four levels up is the project root (motionbricks package root).
        _pkg_root = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.normpath(os.path.join(_pkg_root, "..", "..", ".."))

        def _default(attr: str, *path_parts: str) -> None:
            if not getattr(self.args, attr, None):
                setattr(self.args, attr, os.path.abspath(os.path.join(*path_parts)))

        _default("result_dir", _project_root, "out")
        _default("data_root", _project_root, "datasets")
        _default("humanoid_scene_xml", _project_root, "assets", "skeletons", "g1", "scene_29dof.xml")
        _default("skeleton_xml", _project_root, "assets", "skeletons", "g1", "g1.xml")
        _default("explicit_dataset_folder", _project_root, "datasets", "motionbricks-G1")

        # clips_ckpt: derive from result_dir if not set
        if not getattr(self.args, "clips_ckpt", None):
            self.args.clips_ckpt = os.path.abspath(
                os.path.join(self.args.result_dir, "G1-clip.ckpt")
            )

    def _load_models(self) -> None:
        """Load pose/root models and optionally the clip bank."""
        self.args.return_model_configs = True
        reprocess = getattr(self.args, "reprocess_clips", False)
        ckpt_exists = os.path.exists(self.args.clips_ckpt)

        if self.args.clips_ckpt is None or not ckpt_exists or reprocess:
            self.args.return_dataloader = True
            models, confs, train_dl, val_dl = test(self.args)
        else:
            self.args.return_dataloader = False
            models, confs = test(self.args)
            train_dl = val_dl = None

        for name in ("pose", "root"):
            state_dict = t.load(confs[name].ckpt_path)["state_dict"]
            models[name].load_state_dict(state_dict)

        self.inferencer = motion_inference(models, models["pose"].args)

        from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent

        speed_scale = (
            getattr(self.args, "speed_scale", [0.8, 1.2])
            if getattr(self.args, "random_speed_scale", False)
            else [1.0, 1.0]
        )
        self.full_agent = full_navigation_agent(
            self.inferencer,
            train_dl,
            device="cuda",
            speed_scale=speed_scale,
            target_root_realignment=getattr(self.args, "target_root_realignment", True),
            source_root_realignment=getattr(self.args, "source_root_realignment", True),
            force_canonicalization=getattr(self.args, "force_canonicalization", True),
            skeleton_xml=self.args.skeleton_xml,
            skip_ending_target_cond=getattr(self.args, "skip_ending_target_cond", False),
            filter_qpos=getattr(self.args, "pre_filter_qpos", True),
            clips=self.args.clips,
            ckpt_path=self.args.clips_ckpt,
            reprocess_clips=reprocess,
            val_dataloader=val_dl,
        ).to("cuda")

    def _build_controller(self) -> None:
        min_tok = self.inferencer._args["min_tokens"]
        max_tok = self.inferencer._args["max_tokens"]

        custom_bank = _load_custom_bank(getattr(self.args, "pingpong_bank_json", None))

        self.controller = PingPongTargetController(
            lookat_movement_direction=getattr(self.args, "lookat_movement_direction", False),
            clips=self.args.clips,
            min_token=min_tok,
            max_token=max_tok,
            target_x_range=getattr(self.args, "pingpong_target_x_range", (-0.45, -0.18)),
            target_y_range=getattr(self.args, "pingpong_target_y_range", (0.85, 1.15)),
            target_z_range=getattr(self.args, "pingpong_target_z_range", (0.35, 0.75)),
            swing_velocity=getattr(self.args, "pingpong_swing_velocity", (-0.20, 0.00, 1.20)),
            default_mode=getattr(self.args, "pingpong_mode", "none"),
            root_step_scale=getattr(self.args, "pingpong_root_step_scale", 1.0),
            root_lateral_distance=getattr(self.args, "pingpong_root_lateral_distance", 0.5),
            bank_source=getattr(self.args, "pingpong_bank_source", "existing"),
            target_tokens=getattr(self.args, "pingpong_target_tokens", 6),
            custom_bank=custom_bank,
        )

    def _build_simulator(self) -> None:
        self.mj_model, self.mj_data = build_mj_simulator(
            self.args.humanoid_scene_xml,
            self.inferencer.motion_rep.fps,
        )
