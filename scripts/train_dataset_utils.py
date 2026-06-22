from __future__ import annotations

from torch.utils.data import ConcatDataset, WeightedRandomSampler
import torch

from motionbricks.data.motion_feature_dataset import MotionFeatureDataset


def make_motion_feature_dataset_and_sampler(train_conf, min_frames: int):
    if train_conf.get("seed_dataset") and train_conf.get("pingpong_dataset"):
        seed_dataset = MotionFeatureDataset(train_conf.seed_dataset, min_frames=min_frames)
        pingpong_dataset = MotionFeatureDataset(train_conf.pingpong_dataset, min_frames=min_frames)
        dataset = ConcatDataset([seed_dataset, pingpong_dataset])

        pingpong_ratio = min(max(float(train_conf.get("pingpong_batch_ratio", 0.5)), 0.0), 1.0)
        weights = torch.cat([
            torch.full([len(seed_dataset)], (1.0 - pingpong_ratio) / max(len(seed_dataset), 1), dtype=torch.double),
            torch.full([len(pingpong_dataset)], pingpong_ratio / max(len(pingpong_dataset), 1), dtype=torch.double),
        ])
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(train_conf.get("samples_per_epoch", len(dataset))),
            replacement=True,
        )
        info = {
            "mode": "mixed",
            "seed_samples": len(seed_dataset),
            "pingpong_samples": len(pingpong_dataset),
            "pingpong_batch_ratio": pingpong_ratio,
        }
        return dataset, sampler, info

    dataset = MotionFeatureDataset(train_conf.dataset, min_frames=min_frames)
    return dataset, None, {"mode": "single", "samples": len(dataset)}
