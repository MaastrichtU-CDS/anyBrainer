"""Utility functions for evaluation."""

__all__ = [
    "top1_accuracy",
]

import torch


def top1_accuracy(logits: torch.Tensor, targets_one_hot: torch.Tensor) -> torch.Tensor:
    """Compute top-1 accuracy."""
    targets_idx = targets_one_hot.argmax(dim=1)
    preds = logits.argmax(dim=1)
    return (preds == targets_idx).float().mean()