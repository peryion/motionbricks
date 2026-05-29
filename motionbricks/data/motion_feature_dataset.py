from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from motionbricks.data.synthetic_dataset import collate_batch, collate_tensors


class MotionFeatureDataset(Dataset):
    """Dataset for precomputed normalized MotionBricks motion features."""

    def __init__(self, path: str | Path, min_frames: int = 1):
        self.path = Path(path)
        state: Any = torch.load(self.path, map_location="cpu")
        if isinstance(state, dict) and "samples" in state:
            samples = state["samples"]
        elif isinstance(state, list):
            samples = state
        else:
            raise ValueError(f"Unsupported dataset format in {self.path}")

        self.samples = []
        for idx, sample in enumerate(samples):
            motion = sample["motion"].float()
            if motion.shape[0] < min_frames:
                continue
            self.samples.append({
                "keyid": sample.get("keyid", idx),
                "motion": motion,
            })
        if not self.samples:
            raise ValueError(f"No samples with at least {min_frames} frames in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

