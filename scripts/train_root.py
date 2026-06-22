"""Root model training script using synthetic data.

Demonstrates how the root backbone training pipeline works without
requiring the actual motion dataset. Loads the saved model config from
the checkpoint directory and trains on randomly generated motion tensors.

The root model does not require a pretrained VQVAE — it directly
predicts continuous root motion values.

Usage:
    python scripts/train_root.py --max_steps 100
"""

import argparse
import copy
import os

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from motionbricks.data.synthetic_dataset import SyntheticMotionDataset, collate_batch
from motionbricks.helper.pl_util import load_motion_rep
from train_common import (
    configure_cuda_process,
    load_matching_checkpoint,
    load_training_config,
    make_loggers_and_callbacks,
    make_trainer,
)
from train_dataset_utils import make_motion_feature_dataset_and_sampler


def load_config(result_dir: str, train_conf: DictConfig):
    """Load and patch hparams.yaml for single-GPU training."""
    version_dir = os.path.join(result_dir, "motionbricks_root", "version_1")
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    conf = OmegaConf.load(hparams_path)

    with open_dict(conf):
        # resolve data paths to the version directory (where skeleton/stats live)
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")

        # single-GPU training overrides
        conf.trainer.devices = train_conf.trainer.devices
        conf.trainer.num_nodes = train_conf.trainer.num_nodes
        conf.trainer.max_steps = train_conf.max_steps
        conf.trainer.accelerator = train_conf.trainer.accelerator
        conf.trainer.strategy = train_conf.trainer.strategy
        conf.trainer.enable_progress_bar = train_conf.trainer.enable_progress_bar
        conf.trainer.log_every_n_steps = train_conf.trainer.log_every_n_steps
        conf.trainer.val_check_interval = train_conf.trainer.val_check_interval
        conf.trainer.num_sanity_val_steps = 0

        # resolve ${trainer.max_steps} in scheduler
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

        # remove keys with unresolvable ${hydra:...} interpolations
        conf.id = "synthetic"
        conf.run_dir = "."
        conf.out_dir = result_dir

    # resolve all ${} interpolations, then re-wrap as DictConfig
    resolved = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(resolved)

    return conf, version_dir


def main():
    parser = argparse.ArgumentParser(description="Root model training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with training parameters")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Directory containing pretrained checkpoints")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of synthetic samples in dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Precomputed MotionBricks motion feature dataset .pt")
    parser.add_argument("--min_tokens", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--wandb", type=int, default=None, help="1 enables wandb, 0 disables it")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_offline", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cli_overrides = {
        "result_dir": args.result_dir,
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_samples": args.num_samples,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "log_dir": args.log_dir,
        "run_name": args.run_name,
        "seed": args.seed,
    }
    train_conf = load_training_config(args.config, cli_overrides, "root")
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
    pl.seed_everything(train_conf.seed)
    conf, version_dir = load_config(train_conf.result_dir, train_conf)

    # instantiate skeleton and motion representation
    motion_rep = load_motion_rep(conf)
    feat_dim = len(motion_rep.indices['all'])

    if train_conf.dataset or (train_conf.get("seed_dataset") and train_conf.get("pingpong_dataset")):
        min_frames = conf.model.args.min_tokens * (2 ** conf.model.args.down_t) + 1
        dataset, sampler, dataset_info = make_motion_feature_dataset_and_sampler(train_conf, min_frames)
    else:
        dataset = SyntheticMotionDataset(
            feat_dim=feat_dim,
            num_samples=train_conf.num_samples,
            min_frames=200,
            max_frames=400,
        )
        sampler = None
        dataset_info = {"mode": "synthetic", "samples": len(dataset)}
    dataloader = DataLoader(
        dataset,
        batch_size=train_conf.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=train_conf.num_workers,
        collate_fn=collate_batch,
        persistent_workers=train_conf.num_workers > 0,
    )

    # instantiate networks and model
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        # instantiate backbone network (needs full motion_rep for dual_rep access)
        backbone_net = instantiate(
            model_conf.backbone_network,
            motion_rep=motion_rep,
            _recursive_=False,
        )

        # build optimizer and scheduler as partials
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None

        model = instantiate(
            model_conf,
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone_net,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    if train_conf.init_from_checkpoint.enabled:
        init_ckpt = train_conf.init_from_checkpoint.path
        if init_ckpt is None:
            init_ckpt = os.path.join(version_dir, "checkpoints", "model-step=2000000.ckpt")
        load_matching_checkpoint(model, init_ckpt, strict=bool(train_conf.init_from_checkpoint.strict))

    logger, callbacks, run_dir = make_loggers_and_callbacks(train_conf, conf, "root")
    trainer = make_trainer(train_conf, conf, logger, callbacks)

    print(f"Starting root model training for {train_conf.max_steps} steps...")
    print(f"  Run dir: {run_dir}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Batch size: {train_conf.batch_size}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Dataset info: {dataset_info}")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Training complete.")


if __name__ == "__main__":
    main()
