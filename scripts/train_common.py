from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


DEFAULT_TRAIN_CONFIG = {
    "result_dir": "./out",
    "dataset": None,
    "max_steps": 200,
    "batch_size": 8,
    "num_workers": 2,
    "num_samples": 500,
    "min_tokens": None,
    "max_tokens": None,
    "seed": 42,
    "log_dir": "out/training_logs",
    "run_name": None,
    "trainer": {
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        "strategy": "auto",
        "matmul_precision": "high",
        "enable_progress_bar": True,
        "log_every_n_steps": 10,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "limit_val_batches": 0.0,
        "num_sanity_val_steps": 0,
    },
    "wandb": {
        "enabled": True,
        "project": "motionbricks",
        "entity": None,
        "offline": False,
        "tags": [],
    },
    "checkpoint": {
        "enabled": True,
        "filename": "{step}",
        "every_n_train_steps": 1000,
        "save_last": True,
        "save_top_k": -1,
        "save_weights_only": False,
    },
    "learning_rate_monitor": {
        "enabled": True,
        "logging_interval": "step",
    },
    "init_from_checkpoint": {
        "enabled": True,
        "path": None,
        "strict": False,
    },
}


def load_training_config(path: str | None, cli_overrides: dict[str, Any], model_name: str) -> DictConfig:
    conf = OmegaConf.create(DEFAULT_TRAIN_CONFIG)
    if path:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))
    conf = OmegaConf.merge(conf, {k: v for k, v in cli_overrides.items() if v is not None})
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if conf.run_name is None:
        conf.run_name = model_name
    conf.run_name = f"{conf.run_name}-{stamp}"
    return conf


def make_loggers_and_callbacks(train_conf: DictConfig, model_conf: DictConfig, model_name: str):
    run_dir = Path(train_conf.log_dir) / train_conf.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved = OmegaConf.create({
        "train": OmegaConf.to_container(train_conf, resolve=True),
        "model": OmegaConf.to_container(model_conf, resolve=True),
    })
    OmegaConf.save(resolved, run_dir / "config.yaml")

    logger = False
    if train_conf.wandb.enabled:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb logging is enabled, but wandb is not installed. "
                "Install it with `pip install wandb`, or set `wandb.enabled: false` in the train config."
            ) from exc
        logger = WandbLogger(
            project=train_conf.wandb.project,
            entity=train_conf.wandb.entity,
            name=train_conf.run_name,
            save_dir=str(run_dir),
            offline=bool(train_conf.wandb.offline),
            tags=list(train_conf.wandb.tags),
            config=OmegaConf.to_container(resolved, resolve=True),
        )

    callbacks = []
    if train_conf.learning_rate_monitor.enabled:
        callbacks.append(LearningRateMonitor(logging_interval=train_conf.learning_rate_monitor.logging_interval))
    if train_conf.checkpoint.enabled:
        callbacks.append(ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=train_conf.checkpoint.filename,
            every_n_train_steps=train_conf.checkpoint.every_n_train_steps,
            save_last=train_conf.checkpoint.save_last,
            save_top_k=train_conf.checkpoint.save_top_k,
            save_weights_only=train_conf.checkpoint.save_weights_only,
        ))

    return logger, callbacks, run_dir


def make_trainer(train_conf: DictConfig, model_conf: DictConfig, logger, callbacks) -> pl.Trainer:
    import torch

    if train_conf.trainer.matmul_precision:
        torch.set_float32_matmul_precision(train_conf.trainer.matmul_precision)

    return pl.Trainer(
        max_steps=train_conf.max_steps,
        devices=train_conf.trainer.devices,
        num_nodes=train_conf.trainer.num_nodes,
        accelerator=train_conf.trainer.accelerator,
        strategy=train_conf.trainer.strategy,
        precision=model_conf.trainer.precision,
        gradient_clip_val=model_conf.trainer.gradient_clip_val,
        enable_progress_bar=train_conf.trainer.enable_progress_bar,
        log_every_n_steps=train_conf.trainer.log_every_n_steps,
        val_check_interval=train_conf.trainer.val_check_interval,
        check_val_every_n_epoch=train_conf.trainer.check_val_every_n_epoch,
        limit_val_batches=train_conf.trainer.limit_val_batches,
        num_sanity_val_steps=train_conf.trainer.num_sanity_val_steps,
        enable_checkpointing=bool(train_conf.checkpoint.enabled),
        logger=logger,
        callbacks=callbacks,
    )


def load_matching_checkpoint(model, ckpt_path: str | Path, strict: bool = False):
    import torch

    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    matched = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        matched[key] = value

    missing, unexpected = model.load_state_dict(matched, strict=False)
    if strict and (skipped_shape or skipped_missing or unexpected or missing):
        raise RuntimeError(
            f"Strict checkpoint load failed for {ckpt_path}: "
            f"matched={len(matched)}, skipped_shape={len(skipped_shape)}, "
            f"skipped_missing={len(skipped_missing)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    print(f"Loaded partial checkpoint: {ckpt_path}")
    print(f"  matched tensors: {len(matched)}")
    print(f"  skipped shape mismatches: {len(skipped_shape)}")
    for key, src_shape, dst_shape in skipped_shape[:20]:
        print(f"    {key}: {src_shape} -> {dst_shape}")
    if len(skipped_shape) > 20:
        print(f"    ... {len(skipped_shape) - 20} more")
    print(f"  skipped source-only tensors: {len(skipped_missing)}")
    print(f"  current missing tensors after partial load: {len(missing)}")
    return {
        "matched": len(matched),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "missing": missing,
        "unexpected": unexpected,
    }
