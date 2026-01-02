"""MIL aggregation modes."""

__all__ = [
    "noisy_or_bag_logits",
    "lse_bag_logits_from_instance_logits",
    "topk_mean_bag_logits",
]

import torch


def _log1mexp(a: torch.Tensor) -> torch.Tensor:
    # stable log(1 - exp(a)) for a <= 0
    # split at log(0.5) ~ -0.693 for best stability
    out = torch.empty_like(a)
    mask = a < -0.6931471805599453
    out[mask] = torch.log1p(-torch.exp(a[mask]))
    out[~mask] = torch.log(-torch.expm1(a[~mask]))
    return out


def noisy_or_bag_logits(z: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """`Noisy OR` cell-level bagging from feature maps.

    Feature maps are assumed to be condensed to `n_classes` channels.

    Args:
        z: Logits of shape (B, n_classes, *spatial_dims).
        dims: Dimensions to sum over.

    Returns:
        bag-level logits, shape (B, C)
    """
    # For each instance: 1 - p_i = sigmoid(-z_i); log(1 - p_i) = -softplus(z_i)
    log_q = -torch.nn.functional.softplus(z)  # same shape as z
    log_q_sum = log_q.sum(dim=dims)  # (B, C)
    # p_bag = 1 - exp(log_q_sum); bag_logit = log(p_bag/(1-p_bag))
    return _log1mexp(log_q_sum) - log_q_sum


def lse_bag_logits_from_instance_logits(
    z: torch.Tensor, dims: tuple[int, ...], tau: float = 1.0
) -> torch.Tensor:
    """LogSumExp pooling on logits (soft-max pooling). Returns bag logits (B,
    C).

    Args:
        z: Logits of shape (B, n_classes, *spatial_dims).
        dims: Dimensions to sum over.
        tau: Temperature parameter for the softmax.

    Returns:
        bag-level logits, shape (B, C)
    """
    n = 1
    for d in dims:
        n *= z.shape[d]
    # τ*log(mean exp(z/τ)) = τ*(logsumexp(z/τ) - log N)
    return tau * (
        torch.logsumexp(z / tau, dim=dims)
        - torch.log(torch.tensor(float(n), device=z.device))  # (B, C)
    )


def topk_mean_bag_logits(
    z: torch.Tensor, dims: tuple[int, ...], k: int = 8
) -> torch.Tensor:
    """Average the top-k instance logits per class. Returns bag logits (B, C).

    Args:
        z: Logits of shape (B, n_classes, *spatial_dims).
        dims: Dimensions to sum over.
        k: Number of top-k instances to average.

    Returns:
        bag-level logits, shape (B, C)
    """
    B, C = z.shape[:2]
    flat = z.flatten(start_dim=2)  # (B, C, N)
    k = min(k, flat.shape[-1])
    topk_vals, _ = torch.topk(flat, k, dim=-1)
    return topk_vals.mean(dim=-1)  # (B, C)
