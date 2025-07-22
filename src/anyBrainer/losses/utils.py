"""Utility functions for computing contrastive learning statistics."""

__all__ = [
    "compute_cl_stats",
]

import torch
import torch.nn.functional as F


def compute_cl_stats(
    logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute contrastive learning statistics.

    Args:
        logits: (B, 1 + K) - logits from the model (1 positive, K negatives)
        temperature: float - temperature for the softmax function

    Returns:
        dict[str, torch.Tensor] - dictionary with the statistics
    """
    with torch.no_grad():
        # Mean negative similarity
        finite_neg = torch.isfinite(logits[:, 1:]) # Only consider finite negatives (after masking)
        neg_mean = (
            logits[:, 1:][finite_neg].mean()
            if finite_neg.any()
            else torch.tensor(float("nan"), device=logits.device)
        )
        # Contrastive accuracy
        contrastive_acc = (logits.argmax(dim=1) == 0).float().mean()

        # Negative entropy
        neg_probs = F.softmax(logits[:, 1:], dim=1)
        neg_entropy = -torch.sum(neg_probs * neg_probs.clamp_min(1e-12).log(), dim=1)

    return {
        "pos_mean": logits[:, 0].mean().detach(),
        "neg_mean": neg_mean.detach(),
        "contrastive_acc": contrastive_acc.detach(),
        "neg_entropy": neg_entropy.mean().detach(),
    }