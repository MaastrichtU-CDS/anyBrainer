"""Utility functions for creating and running models."""

import logging

import torch

logger = logging.getLogger(__name__)

def sync_dist_safe(obj) -> bool:
    """
    Return True if the attached Trainer is in DDP/FSDP mode (world_size > 1).

    Works for LightningModule, Callback, or anything else that has a .trainer.
    Falls back to False if we cannot decide.
    """
    trainer = getattr(obj, "trainer", None)
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
