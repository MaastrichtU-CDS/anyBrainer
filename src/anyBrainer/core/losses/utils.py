"""Utility functions for computing contrastive learning statistics."""

__all__ = [
    "compute_cl_stats",
]

import torch

from anyBrainer.registry import register, RegistryKind as RK


@register(RK.UTIL)
def compute_cl_stats(
    logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute contrastive learning statistics.

    Args:
        logits: (B, 1 + K) - logits from the model (1 positive, K negatives)
        temperature: float - temperature for the softmax function

    Returns:
        dict[str, torch.Tensor] - dictionary with the statistics
    """
    with torch.no_grad():
        # Mean negative similarity
        finite_neg = torch.isfinite(
            logits[:, 1:]
        )  # Only consider finite negatives (after masking)
        neg_mean = (
            logits[:, 1:][finite_neg].mean()
            if finite_neg.any()
            else torch.tensor(float("nan"), device=logits.device)
        )
        # Contrastive accuracy
        contrastive_acc = (logits.argmax(dim=1) == 0).float().mean()

    return {
        "pos_mean": logits[:, 0].mean().detach(),
        "neg_mean": neg_mean.detach(),
        "contrastive_acc": contrastive_acc.detach(),
    }
