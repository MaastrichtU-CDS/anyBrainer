"""Helper functions for feature visualization."""

from __future__ import annotations

__all__ = [
    "pca_to_3ch_volume",
    "rescale_per_ch_feats",
    "upsample_to_target_grid",
    "concat_multilevel_feats",
    "normalize_level_for_concat",
]

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def pca_to_3ch_volume(
    feat: torch.Tensor,
    sample_voxels: int = 100_000,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PCA is fit on a random subset of voxels, then applied to all voxels.

    Args:
        feat: Tensor of shape (C, D, H, W).
        sample_voxels: number of voxels to sample for PCA
        eps: epsilon for numerical stability

    Returns:
       PCA-projected volume of shape (3, D, H, W).
    """
    if feat.ndim != 4:
        msg = f"Expected (C, D, H, W), got {feat.shape}"
        logger.error(msg)
        raise ValueError(msg)

    C, D, H, W = feat.shape
    N = D * H * W

    # X: (N, C)
    X = feat.reshape(C, N).T.contiguous()

    # Center per channel
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    # Sample voxels to fit PCA basis
    M = max(min(sample_voxels, N), 1)
    idx = torch.randperm(N, device=X.device)[:M]
    Xs = Xc[idx]  # (M, C)

    # Standardize channels
    std = Xs.std(dim=0, keepdim=True).clamp_min(eps)
    Xs = Xs / std
    Xc = Xc / std

    # PCA low-rank (top 3)
    # Xs is approximately equal to U S V^T, where V is (C, 3)
    _, _, V = torch.pca_lowrank(Xs, q=3, center=False)
    V3 = V[:, :3]  # (C, 3)

    # Project all voxels
    Y = Xc @ V3  # (N, 3)
    Y3 = Y.T.reshape(3, D, H, W).contiguous()  # (3, D, H, W)
    return Y3


@torch.no_grad()
def rescale_per_ch_feats(vol: torch.Tensor, lo=1.0, hi=99.0) -> torch.Tensor:
    """Rescales each channel independently to [0,1] using percentiles.

    vol: (C, D, H, W) or (D, H, W)
    """
    if vol.ndim == 3:
        v = vol.flatten()
        a = torch.quantile(v, lo / 100.0)
        b = torch.quantile(v, hi / 100.0)
        return ((vol - a) / (b - a + 1e-6)).clamp(0, 1)

    C = vol.shape[0]
    out = []
    for c in range(C):
        out.append(rescale_per_ch_feats(vol[c]))
    return torch.stack(out, dim=0)


@torch.no_grad()
def upsample_to_target_grid(x: torch.Tensor, target_shape) -> torch.Tensor:
    """
    x: (C, d, h, w)
    target_shape: (D, H, W)
    returns: (C, D, H, W)
    """
    x = x.unsqueeze(0)  # (1, C, d, h, w)
    x_up = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    return x_up.squeeze(0)


@torch.no_grad()
def concat_multilevel_feats(
    feats: list[torch.Tensor], target_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Concatenate multi-level features into a single volume.

    feats: list of (C_l, d_l, h_l, w_l)
    target_shape: (D, H, W)
    returns: (sum C_l, D, H, W)
    """
    feats_up = [
        upsample_to_target_grid(f, target_shape) for f in feats
    ]  # each (C_l, D, H, W)
    return torch.cat(feats_up, dim=0)  # (sum C_l, D, H, W)


@torch.no_grad()
def normalize_level_for_concat(f_up: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    f_up: (C, D, H, W) already on target grid.
    Returns: (C, D, H, W) with per-channel z-score + per-level energy normalization.
    """
    if f_up.ndim != 4:
        msg = f"Expected (C,D,H,W), got {f_up.shape}"
        logger.error(msg)
        raise ValueError(msg)

    C, D, H, W = f_up.shape
    N = D * H * W

    X = f_up.reshape(C, N).T.contiguous()  # (N, C)

    # Channel-wise z-score (correlation-PCA effect)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(eps)
    X = (X - mu) / sd

    # Per-level energy normalization (equalize average squared L2 norm per voxel)
    v = (X.pow(2).sum(dim=1).mean()).clamp_min(eps)  # scalar
    X = X / v.sqrt()

    return X.T.reshape(C, D, H, W).contiguous()
