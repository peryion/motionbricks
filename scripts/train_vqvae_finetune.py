"""Fine-tune the pose VQ-VAE on precomputed MotionBricks feature datasets."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from motionbricks.data.motion_feature_dataset import MotionFeatureDataset
from motionbricks.data.synthetic_dataset import collate_batch
from motionbricks.helper.pl_util import load_motion_rep
from train_common import (
    configure_cuda_process,
    load_matching_checkpoint,
    load_training_config,
    make_loggers_and_callbacks,
    make_trainer,
    patch_torch_load_map_location_cpu,
)


def _load_model_config(result_dir: str, train_conf: DictConfig):
    version_dir = os.path.join(result_dir, "motionbricks_vqvae", "version_1")
    conf = OmegaConf.load(os.path.join(version_dir, "hparams.yaml"))

    with open_dict(conf):
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")

        conf.trainer.devices = train_conf.trainer.devices
        conf.trainer.num_nodes = train_conf.trainer.num_nodes
        conf.trainer.max_steps = train_conf.max_steps
        conf.trainer.accelerator = train_conf.trainer.accelerator
        conf.trainer.strategy = train_conf.trainer.strategy
        conf.trainer.enable_progress_bar = train_conf.trainer.enable_progress_bar
        conf.trainer.log_every_n_steps = train_conf.trainer.log_every_n_steps
        conf.trainer.val_check_interval = train_conf.trainer.val_check_interval
        conf.trainer.num_sanity_val_steps = 0

        conf.model.scheduler.num_training_steps = train_conf.max_steps
        if "lr" in train_conf and train_conf.lr is not None:
            conf.model.optimizer.lr = train_conf.lr
        if "final_lr" in train_conf and train_conf.final_lr is not None:
            conf.model.scheduler.final_lr = train_conf.final_lr
        if "warmup_steps" in train_conf and train_conf.warmup_steps is not None:
            conf.model.scheduler.num_warmup_steps = train_conf.warmup_steps
        if train_conf.min_tokens is not None:
            conf.model.args.min_tokens = train_conf.min_tokens
        if train_conf.max_tokens is not None:
            conf.model.args.max_tokens = train_conf.max_tokens

        conf.id = "vqvae_finetune"
        conf.run_dir = "."
        conf.out_dir = result_dir

    return OmegaConf.create(OmegaConf.to_container(conf, resolve=True)), version_dir


def _make_mixed_dataset(train_conf: DictConfig, min_frames: int):
    seed_dataset = MotionFeatureDataset(train_conf.seed_dataset, min_frames=min_frames)
    pingpong_dataset = MotionFeatureDataset(train_conf.pingpong_dataset, min_frames=min_frames)
    dataset = ConcatDataset([seed_dataset, pingpong_dataset])

    pingpong_ratio = float(train_conf.get("pingpong_batch_ratio", 0.5))
    pingpong_ratio = min(max(pingpong_ratio, 0.0), 1.0)
    seed_weight = (1.0 - pingpong_ratio) / max(len(seed_dataset), 1)
    pingpong_weight = pingpong_ratio / max(len(pingpong_dataset), 1)
    weights = torch.cat([
        torch.full([len(seed_dataset)], seed_weight, dtype=torch.double),
        torch.full([len(pingpong_dataset)], pingpong_weight, dtype=torch.double),
    ])

    num_samples = int(train_conf.get("samples_per_epoch", len(dataset)))
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return dataset, sampler, len(seed_dataset), len(pingpong_dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pose VQ-VAE on SEED + pingpong features")
    parser.add_argument("--config", type=str, default="configs/train_gmr_pingpong_vqvae_finetune.yaml")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--min_tokens", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--wandb", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_offline", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cli_overrides = {
        "result_dir": args.result_dir,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "log_dir": args.log_dir,
        "run_name": args.run_name,
        "seed": args.seed,
    }
    train_conf = load_training_config(args.config, cli_overrides, "vqvae-pingpong-ft")
    if args.devices is not None:
        train_conf.trainer.devices = args.devices
    if args.num_nodes is not None:
        train_conf.trainer.num_nodes = args.num_nodes
    if args.accelerator is not None:
        train_conf.trainer.accelerator = args.accelerator
    if args.strategy is not None:
        train_conf.trainer.strategy = args.strategy
    if args.wandb is not None:
        train_conf.wandb.enabled = bool(args.wandb)
    if args.wandb_project is not None:
        train_conf.wandb.project = args.wandb_project
    if args.wandb_offline is not None:
        train_conf.wandb.offline = bool(args.wandb_offline)
    if args.checkpoint_every is not None:
        train_conf.checkpoint.every_n_train_steps = args.checkpoint_every
    if not train_conf.wandb.enabled:
        train_conf.learning_rate_monitor.enabled = False

    configure_cuda_process(train_conf)
    patch_torch_load_map_location_cpu()
    pl.seed_everything(train_conf.seed)
    conf, version_dir = _load_model_config(train_conf.result_dir, train_conf)

    motion_rep = load_motion_rep(conf)
    min_frames = conf.model.args.min_tokens * (2 ** conf.model.args.down_t) + 1
    dataset, sampler, seed_len, pingpong_len = _make_mixed_dataset(train_conf, min_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=train_conf.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=train_conf.num_workers,
        collate_fn=collate_batch,
        persistent_workers=train_conf.num_workers > 0,
    )

    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        pose_net = instantiate(
            model_conf.pose_vqvae_network,
            motion_rep=motion_rep.dual_rep.local_motion_rep,
        )
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
        model = instantiate(
            model_conf,
            pose_vqvae_network=pose_net,
            root_vqvae_network=None,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    init_ckpt = train_conf.init_from_checkpoint.path
    if init_ckpt is None:
        init_ckpt = os.path.join(version_dir, "checkpoints", "model-step=2000000.ckpt")
    if train_conf.init_from_checkpoint.enabled:
        load_matching_checkpoint(model, init_ckpt, strict=bool(train_conf.init_from_checkpoint.strict))

    logger, callbacks, run_dir = make_loggers_and_callbacks(train_conf, conf, "vqvae")
    trainer = make_trainer(train_conf, conf, logger, callbacks)

    print(f"Starting VQ-VAE fine-tuning for {train_conf.max_steps} steps...")
    print(f"  Run dir: {run_dir}")
    print(f"  Init checkpoint: {Path(init_ckpt)}")
    print(f"  SEED samples: {seed_len}")
    print(f"  Pingpong samples: {pingpong_len}")
    print(f"  Pingpong batch ratio: {float(train_conf.pingpong_batch_ratio):.2f}")
    print(f"  Batch size: {train_conf.batch_size}")
    print(f"  LR: {conf.model.optimizer.lr}")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
