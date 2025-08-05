"""Utility functions for creating and running models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, cast, TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from anyBrainer.registry import flush, get  
from anyBrainer.registry import RegistryKind as RK
from anyBrainer.interfaces import PLModuleMixin

if TYPE_CHECKING:
    import lightning.pytorch as pl

logger = logging.getLogger(__name__)

def sync_dist_safe(obj) -> bool:
    """
    Return True if the attached Trainer is in DDP/FSDP mode (world_size > 1).

    Works for LightningModule, Callback, or anything else that has a .trainer.
    Falls back to False if we cannot decide.
    """
    try:
        trainer = getattr(obj, "trainer", None)
    except RuntimeError:
        return False
    
    if trainer is None:
        return False

    try:
        return getattr(trainer, "world_size", 1) > 1
    except AttributeError:
        return False

def get_sub_ses_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get subject and session ID tensors from batch."""
    if "sub_id" not in batch or "ses_id" not in batch:
        msg = "sub_id and ses_id must be in batch"
        logger.error(msg)
        raise ValueError(msg)

    def parse_id(value):
        if isinstance(value, list):
            return torch.tensor([int(str(x).split("_")[-1]) for x in value], device=device)
        elif isinstance(value, torch.Tensor):
            return value.to(device)
        else:
            msg = f"Expected list or tensor, but got {type(value)}"
            logger.error(msg)
            raise ValueError(msg)

    sub_id = parse_id(batch["sub_id"])
    ses_id = parse_id(batch["ses_id"])

    return sub_id, ses_id

def pack_ids(subject_ids: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
    """
    Pack subject and session IDs into a single tensor.
    Assumes both < 1e6; change multiplier if needed.
    """
    return subject_ids.long() * 1_000_000 + session_ids.long()

def unpack_settings_for_train_workflow(
    global_settings: dict[str, Any],
    logging_settings: dict[str, Any],
    pl_datamodule_settings: dict[str, Any],
    pl_module_settings: dict[str, dict[str, Any]],
    pl_callback_settings: list[dict[str, Any]],
    pl_trainer_settings: dict[str, Any],
    ckpt_settings: dict[str, Any],
) -> dict[str, Any]: 
    """Unpack user-provided settings and provide default values for a typical train workflow."""
    return {
        "project": global_settings.get("project", "anyBrainer"),
        "experiment": global_settings.get("experiment", "exp-01"),
        "root_dir": global_settings.get("root_dir", Path.cwd()),
        "seed": global_settings.get("seed", 12345),
        "worker_logs": logging_settings.get("worker_logs", True),
        "save_logs": logging_settings.get("save_logs", True),
        "dev_mode": logging_settings.get("dev_mode", False),
        "enable_wandb": logging_settings.get("wandb_enable", True),
        "wandb_watch_enable": logging_settings.get("wandb_watch_enable", False),
        "wandb_watch_kwargs": logging_settings.get("wandb_watch_kwargs", {}),
        "pl_datamodule_name": pl_datamodule_settings.get("name", "ContrastiveDataModule"),
        "data_dir": pl_datamodule_settings.get("data_dir", Path.cwd()),
        "data_handler_kwargs": pl_datamodule_settings.get("data_handler_kwargs", {}),
        "num_workers": pl_datamodule_settings.get("num_workers", 32),
        "batch_size": pl_datamodule_settings.get("batch_size", 8),
        "extra_dataloader_kwargs": pl_datamodule_settings.get("extra_dataloader_kwargs", {}),
        "train_val_test_split": pl_datamodule_settings.get("train_val_test_split", (0.7, 0.15, 0.15)),
        "train_transforms": pl_datamodule_settings.get("train_transforms"),
        "val_transforms": pl_datamodule_settings.get("val_transforms"),
        "test_transforms": pl_datamodule_settings.get("test_transforms"),
        "predict_transforms": pl_datamodule_settings.get("predict_transforms"),
        "new_version": ckpt_settings.get("new_version", False),
        "model_checkpoint": ckpt_settings.get("model_checkpoint"),
        "save_every_n_epochs": ckpt_settings.get("save_every_n_epochs", 1),
        "save_last": ckpt_settings.get("save_last", True),
        "save_top_k": ckpt_settings.get("save_top_k", -1),
        "extra_ckpt_kwargs": ckpt_settings.get("extra_kwargs", {}),
        "pl_module_name": pl_module_settings.get("name", "CLwAuxModel"),
        "pl_module_kwargs": pl_module_settings,
        "pl_callback_kwargs": pl_callback_settings,
        "pl_trainer_kwargs": pl_trainer_settings,
    }

def dict_get_as_tensor(value: Any) -> torch.Tensor | None:
    """Get a value from dict.get() and convert it to a tensor."""
    if value is None:
        return None
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        return value
        
    msg = f"Value must be a list, tensor, or None, but got {type(value)}"
    logger.error(msg)
    raise ValueError(msg)

def resolve_fn(
    fn: Callable | str | None,
) -> Callable | None:
    """Get function from config."""
    if isinstance(fn, str):
        return cast(Callable, get(RK.UTIL, fn))
    return fn

def setup_all_mixins(module: pl.LightningModule, **cfg) -> None:
    """Setup all registered mixins for a PL module."""
    mixins = flush(RK.PL_MODULE_MIXIN)
    for cls in module.__class__.__mro__:
        if cls.__name__ in mixins:
            try:
                cls.setup_mixin(cast(PLModuleMixin, module), **cfg)
            except TypeError:
                pass

def get_ckpt_callback(trainer: pl.Trainer) -> ModelCheckpoint | None:
    """Get the checkpoint callback from the trainer."""
    if not hasattr(trainer, "callbacks"):
        return None

    for cb in trainer.callbacks: # type: ignore[attr-defined]
        if isinstance(cb, ModelCheckpoint):
            return cast(ModelCheckpoint, cb)
    return None