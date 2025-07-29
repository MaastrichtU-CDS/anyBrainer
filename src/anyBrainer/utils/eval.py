"""Utility functions for evaluation."""

__all__ = [
    "top1_accuracy",
]

import torch

from anyBrainer.registry import register, RegistryKind as RK


@register(RK.UTIL)
def top1_accuracy(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    is_one_hot: bool = False,
) -> torch.Tensor:
    """Compute top-1 accuracy."""
    if is_one_hot:
        targets_idx = targets.argmax(dim=1)
    else:
        targets_idx = targets
    preds = logits.argmax(dim=1)
    return (preds == targets_idx).float().mean()