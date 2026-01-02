"""Utility functions for evaluation."""

__all__ = [
    "top1_accuracy",
    "effective_rank",
    "feature_variance",
    "mse_score",
    "rmse_score",
    "mae_score",
    "r2_score",
    "pearsonr",
]

import torch
import logging

from anyBrainer.registry import register, RegistryKind as RK

logger = logging.getLogger(__name__)


def _check_mismatch(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Check if the shape of pred and target are the same."""
    if pred.shape != target.shape:
        msg = (
            f"Shape mismatch between `pred`: {pred.shape} and `target`: {target.shape}"
        )
        logger.error(msg)
        raise ValueError(msg)


@register(RK.UTIL)
def top1_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_one_hot: bool = False,
    is_logits: bool = False,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Top-1 accuracy for binary or multiclass.

    logits: (B,) or (B,1) for binary; (B,C) for multiclass.
    targets: class indices (B,) or one-hot (B,C) if is_one_hot=True.
    is_logits: set to True if `logits` are raw model outputs; set to
        False if `logits` are post-processed (e.g. sigmoided, discretized)
    threshold: used if binary; for `is_logits==True`, only applied when != 0.5
    """
    # binary: shape (B,) or (B,1)
    if logits.ndim == 1 or logits.size(1) == 1:
        x = logits.view(-1)
        if is_logits:
            if threshold == 0.5:
                preds = (x >= 0).long()  # sigmoid(x)>=0.5 <=> x>=0
            else:
                preds = (x.sigmoid() >= threshold).long()
        else:
            # if probs in [0,1], threshold; if already discrete, this still works
            preds = (x >= threshold).long()
        t = targets.view(-1).long()

    # multiclass: shape (B,C)
    else:
        preds = logits.argmax(dim=1)
        t = targets.argmax(dim=1) if is_one_hot else targets.view(-1).long()

    # safety: device, shape, empty-batch
    t = t.to(preds.device)
    preds = preds.reshape(-1)
    t = t.reshape(-1)
    if preds.numel() != t.numel():
        msg = f"top1_accuracy: shape mismatch preds {preds.shape} vs targets {t.shape}"
        logger.error(msg)
        raise ValueError(msg)
    if preds.numel() == 0:
        logger.warning("top1_accuracy: empty batch")
        return torch.tensor(0.0, device=preds.device)

    return (preds == t).float().mean()


@register(RK.UTIL)
def effective_rank(
    features: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically-stable effective rank based on singular values.

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
    features = features.to(torch.float64)  # promote precision
    features -= features.mean(dim=0, keepdim=True)

    # singular values are eigenvalues of cov
    svals = torch.linalg.svdvals(features)  # shape (min(N, D),)
    svals = svals.clamp_min(eps)

    probs = svals / svals.sum()
    entropy = -(probs * probs.log()).sum()
    eff_r = torch.exp(entropy)

    return (
        eff_r.to(dtype=torch.float32),
        entropy.to(dtype=torch.float32),
    )


@register(RK.UTIL)
def feature_variance(features: torch.Tensor) -> torch.Tensor:
    """Compute variance of a feature matrix.

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


@register(RK.UTIL)
def mse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error score."""
    _check_mismatch(pred, target)
    return torch.mean((pred - target) ** 2)


@register(RK.UTIL)
def rmse_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error score."""
    return torch.sqrt(mse_score(pred, target))


@register(RK.UTIL)
def mae_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error score."""
    _check_mismatch(pred, target)
    return torch.mean(torch.abs(pred - target))


@register(RK.UTIL)
def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R-squared score."""
    _check_mismatch(pred, target)
    var = torch.var(target, unbiased=False)
    if var == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - torch.mean((pred - target) ** 2) / var


@register(RK.UTIL)
def pearsonr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation coefficient."""
    _check_mismatch(pred, target)
    pred = pred - pred.mean()
    target = target - target.mean()
    num = (pred * target).sum()
    den = torch.sqrt((pred**2).sum() * (target**2).sum())
    if den == 0:
        return torch.tensor(0.0, device=pred.device)
    return num / den
