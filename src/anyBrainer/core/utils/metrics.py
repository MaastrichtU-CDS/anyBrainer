"""Utility functions for evaluation."""

__all__ = [
    "top1_accuracy",
    "effective_rank",
    "feature_variance",
]

import torch
import logging

from anyBrainer.registry import register, RegistryKind as RK

logger = logging.getLogger(__name__)

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

@register(RK.UTIL)
def effective_rank(
    features: torch.Tensor, 
    eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically-stable effective rank based on singular values.

    Args:
    - features : Tensor, shape (N, D)
    - eps : float, optional, default=1e-12

    Returns:
    - eff_r : Tensor, shape ()
    - entropy : Tensor, shape ()
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (N, D) got {features.shape}")
    
    if features.shape[0] < 1:
        z = torch.zeros((), device=features.device)
        return z, z

    # centre features, but avoid float32 round-off in small batches
    features = features.to(torch.float64) # promote precision
    features -= features.mean(dim=0, keepdim=True)

    # singular values are eigenvalues of cov
    svals = torch.linalg.svdvals(features) # shape (min(N, D),)
    svals = svals.clamp_min(eps)

    probs   = svals / svals.sum()
    entropy = -(probs * probs.log()).sum()
    eff_r   = torch.exp(entropy)
    
    return (
        eff_r.to(dtype=torch.float32), 
        entropy.to(dtype=torch.float32),
    )

@register(RK.UTIL)
def feature_variance(features: torch.Tensor) -> torch.Tensor:
    """
    Compute variance of a feature matrix.

    Args:
    - features : Tensor, shape (N, D)

    Returns:
    - variance : float
        Mean variance of the features
    """
    if features.ndim != 2:
        msg = f"Expected (N, D) got {features.shape}"
        logger.error(msg)
        raise ValueError(msg)

    if features.shape[0] < 1:
        return torch.tensor(0, device=features.device)

    return features.var(dim=0, unbiased=False).mean()