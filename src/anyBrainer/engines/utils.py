"""Utility functions for creating and running models."""

import logging
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl

from anyBrainer.utils.io import resolve_path

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

    # Subject IDs
    if isinstance(batch["sub_id"], list):
        if not any(isinstance(sub_id, str) and sub_id.startswith("sub-") for sub_id in batch["sub_id"]):
            sub_id = torch.tensor(batch["sub_id"], device=device)
        elif all(isinstance(sub_id, str) and sub_id.startswith("sub") for sub_id in batch["sub_id"]):
            sub_id = torch.tensor([int(sub_id.split("-")[-1]) for sub_id in batch["sub_id"]], device=device)
        else:
            msg = "sub_id must be consistent; either as 'sub-x' or 'x'"
            logger.error(msg)
            raise ValueError(msg)
    elif isinstance(batch["sub_id"], torch.Tensor):
        sub_id = batch["sub_id"]
    else:
        msg = f"sub_id must be a list or tensor, but got {type(batch['sub_id'])}"
        logger.error(msg)
        raise ValueError(msg)

    # Session IDs
    if isinstance(batch["ses_id"], list):
        if not any(isinstance(ses_id, str) and ses_id.startswith("ses-") for ses_id in batch["ses_id"]):
            ses_id = torch.tensor(batch["ses_id"], device=device)
        elif all(isinstance(ses_id, str) and ses_id.startswith("ses") for ses_id in batch["ses_id"]):
            ses_id = torch.tensor([int(ses_id.split("-")[-1]) for ses_id in batch["ses_id"]], device=device)
        else:
            msg = "ses_id must be consistent; either as 'ses-x' or 'x'"
            logger.error(msg)
            raise ValueError(msg)
    elif isinstance(batch["ses_id"], torch.Tensor):
        ses_id = batch["ses_id"]
    else:
        msg = f"ses_id must be a list or tensor, but got {type(batch['ses_id'])}"
        logger.error(msg)
        raise ValueError(msg)

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
    ckpt_settings: dict[str, Any],
    trainer_settings: dict[str, Any],
) -> dict[str, Any]: 
    """Unpack user-provided settings and provide default values for a typical train workflow."""
    return {
        "experiment": global_settings.get("experiment", "exp-01"),
        "save_dir": global_settings.get("save_dir", Path.cwd()),
        "seed": global_settings.get("seed", 12345),
        "worker_logs": logging_settings.get("worker_logs", True),
        "dev_mode": logging_settings.get("dev_mode", False),
        "enable_wandb": logging_settings.get("enable_wandb", True),
        "wandb_project": logging_settings.get("wandb_project", "anyBrainer"),
        "pl_datamodule_name": pl_datamodule_settings.get("name", "ContrastiveDataModule"),
        "data_dir": pl_datamodule_settings.get("data_dir", Path.cwd()),
        "num_workers": pl_datamodule_settings.get("num_workers", 32),
        "batch_size": pl_datamodule_settings.get("batch_size", 8),
        "train_val_test_split": pl_datamodule_settings.get("train_val_test_split", (0.7, 0.15, 0.15)),
        "train_transforms": pl_datamodule_settings.get("train_transforms", None),
        "val_transforms": pl_datamodule_settings.get("val_transforms", None),
        "test_transforms": pl_datamodule_settings.get("test_transforms", None),
        "predict_transforms": pl_datamodule_settings.get("predict_transforms", None),
        "new_version": ckpt_settings.get("new_version", False),
        "model_checkpoint": ckpt_settings.get("model_checkpoint", None),
        "save_every_n_epochs": ckpt_settings.get("save_every_n_epochs", 1),
        "save_last": ckpt_settings.get("save_last", True),
        "pl_module_name": pl_module_settings.get("name", "CLwAuxModel"),
        "pl_module_kwargs": pl_module_settings,
        "pl_callback_kwargs": pl_callback_settings,
        "trainer_kwargs": trainer_settings,
    }