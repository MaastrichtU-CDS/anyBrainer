"""Utility functions for creating and running models."""

import logging

import torch.optim as optim


logger = logging.getLogger(__name__)


def get_optimizer_lr(optimizers: list[optim.Optimizer]) -> dict[str, float]:
    """Get optimizer learning rates."""
    return {
        f"train/lr/opt{i}_group{j}": group["lr"]
        for i, opt in enumerate(optimizers)
        for j, group in enumerate(opt.param_groups)
    }

def sync_dist_safe(obj) -> bool:
    """Return True if a Trainer exists and world_size > 1."""
    trainer = getattr(obj, "Trainer", None)
    return bool(trainer and getattr(trainer, "world_size", 1) > 1)
