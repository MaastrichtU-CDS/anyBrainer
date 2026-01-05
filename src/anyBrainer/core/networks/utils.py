"""Utility functions for creating and running models."""

from typing import Sequence, Any
from itertools import product

import logging

import torch
from torch.nn import functional as F

from anyBrainer.core.utils import ensure_tuple_dim

logger = logging.getLogger(__name__)


def get_mlp_head_args(
    in_dim: int,
    out_dim: int,
    num_hidden_layers: int,
    dropout: float | Sequence[float],
    hidden_dim: int | Sequence[int],
    activation: str | Sequence[str],
    activation_kwargs: dict[str, Any] | Sequence[dict[str, Any]],
) -> tuple[list[float], list[int], list[str], list[dict[str, Any]]]:
    """Validate and normalize MLP head arguments."""

    # Normalize dropout
    if isinstance(dropout, float):
        dropouts = [dropout] * (num_hidden_layers + 1)
    elif isinstance(dropout, Sequence):
        dropouts = list(dropout)
        if len(dropouts) != num_hidden_layers + 1:
            msg = (
                f"Expected {num_hidden_layers + 1} dropout values, got {len(dropouts)}"
            )
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"dropout must be float or sequence of floats, got {type(dropout)}"
        logger.error(msg)
        raise TypeError(msg)

    # Normalize hidden_dim
    if isinstance(hidden_dim, int):
        hidden_dims_core = [hidden_dim] * num_hidden_layers
    elif isinstance(hidden_dim, Sequence):
        hidden_dims_core = list(hidden_dim)
        if len(hidden_dims_core) != num_hidden_layers:
            msg = f"Expected {num_hidden_layers} hidden_dim values, got {len(hidden_dims_core)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"hidden_dim must be int or sequence of ints, got {type(hidden_dim)}"
        logger.error(msg)
        raise TypeError(msg)

    hidden_dims = [in_dim] + hidden_dims_core + [out_dim]

    # Normalize activation
    if isinstance(activation, str):
        activations = [activation] * (num_hidden_layers + 1)
    elif isinstance(activation, Sequence):
        activations = list(activation)
        if len(activations) != num_hidden_layers + 1:
            msg = f"Expected {num_hidden_layers + 1} activation values, got {len(activations)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"activation must be str or sequence of str, got {type(activation)}"
        logger.error(msg)
        raise TypeError(msg)

    # Normalize activation_kwargs
    if isinstance(activation_kwargs, dict):
        activation_kwargs = [activation_kwargs] * (num_hidden_layers + 1)
    elif isinstance(activation_kwargs, Sequence):
        activation_kwargs = list(activation_kwargs)
        if len(activation_kwargs) != num_hidden_layers + 1:
            msg = f"Expected {num_hidden_layers + 1} activation_kwargs values, got {len(activation_kwargs)}"
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"activation_kwargs must be dict or sequence of dict, got {type(activation_kwargs)}"
        logger.error(msg)
        raise TypeError(msg)

    return dropouts, hidden_dims, activations, activation_kwargs


def upsample_to_3d(
    x: torch.Tensor, ref: torch.Tensor, mode: str = "trilinear"
) -> torch.Tensor:
    """Upsample x to the same spatial dimensions as ref."""
    return F.interpolate(x, size=ref.shape[-3:], mode=mode, align_corners=False)


def voxel_shuffle_3d(x: torch.Tensor, r: Sequence[int] | int) -> torch.Tensor:
    """Voxel shuffle operation for 3D tensors.

    Can be used to convert a tensor in the patch grid back to the original
    voxel grid.

    Args:
        x: Input tensor of shape (B, C*r1*r2*r3, D, H, W)
        r: Upscale factors per spatial dim; use int for isotropic

    Returns:
        Output tensor of shape (B, C, D*r1, H*r2, W*r3)
    """
    # x: (B, C*r1*r2*r3, D, H, W) -> (B, C, D*r1, H*r2, W*r3)
    B, C_r1_r2_r3, D, H, W = x.shape
    r1, r2, r3 = ensure_tuple_dim(r, 3)
    assert (
        C_r1_r2_r3 % (r1 * r2 * r3) == 0
    ), "Channel dim must be divisible by up1*up2*up3"
    C = C_r1_r2_r3 // (r1 * r2 * r3)
    x = x.view(B, C, r1, r2, r3, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(B, C, D * r1, H * r2, W * r3)


def merge_mask_nd(
    m: torch.Tensor,
    *,
    mode: str = "and",
) -> torch.Tensor:
    """Mirror MONAI-style PatchMerging for N-D boolean masks.

    Args:
        m: Boolean tensor of shape (B, C, *spatial_dims).
        mode: Aggregation rule over the 2^N window:
            - "and": output masked only if all inputs are masked (shrinks mask)
            - "or" : output masked if any input is masked (expands mask)

    Returns:
        Boolean tensor of shape (B, C, dim1//2, dim2//2, ...).
    """
    if m.dtype != torch.bool:
        m = m.to(torch.bool)

    if m.ndim < 3:
        msg = f"Expected (B, C, *spatial), got {m.shape}."
        logger.error(msg)
        raise ValueError(msg)

    spatial = m.shape[2:]
    ndim = len(spatial)

    # Right-pad odd spatial dims
    pads = []
    for s in reversed(spatial):
        pads.extend([0, s % 2])  # (left=0, right=1 if odd)

    if any(pads):
        m = F.pad(m.float(), pads) > 0.5

    # Collect all 2^ndim sub-samplings
    # Example for 3D: (0,0,0), (1,0,0), ..., (1,1,1)
    slices = []
    for offsets in product((0, 1), repeat=ndim):
        sl = [slice(None), slice(None)]
        for o in offsets:
            sl.append(slice(o, None, 2))
        slices.append(m[tuple(sl)])

    # Aggregate
    out = slices[0]
    if mode == "and":
        for s in slices[1:]:
            out = out & s
        return out

    if mode == "or":
        for s in slices[1:]:
            out = out | s
        return out

    msg = f"Unknown mode: {mode}."
    logger.error(msg)
    raise ValueError(msg)


@torch.no_grad()
def center_pad_mask_like(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Center-crop / symmetric-pad a boolean mask to match the spatial shape of
    x.

    Args:
        m: Boolean tensor of shape (B, C, *spatial_m). True = masked.
        x: Tensor of shape (B, C, *spatial_x).

    Returns:
        m_out: Boolean tensor of shape (B, C, *spatial_x).
            Padding is added as False (unmasked) across all channels.
    """
    if m.dtype != torch.bool:
        m = m.to(torch.bool)

    if m.ndim < 3 or x.ndim != m.ndim:
        msg = (
            f"`m` must have at least 3 dimensions and match `x.ndim`; "
            f"got m={m.shape} and x={x.shape}."
        )
        logger.error(msg)
        raise ValueError(msg)

    B_m, C_m, *spatial_m = m.shape
    B_x, C_x, *spatial_x = x.shape

    if B_m != B_x or C_m != C_x:
        msg = (
            f"Batch/Channel mismatch: `m` has (B,C)=({B_m},{C_m}) and "
            f"`x` has (B,C)=({B_x},{C_x}). Expected identical B and C."
        )
        logger.error(msg)
        raise ValueError(msg)

    if len(spatial_m) != len(spatial_x):
        msg = f"Spatial dims mismatch: `m` has {len(spatial_m)} and `x` has {len(spatial_x)}."
        logger.error(msg)
        raise ValueError(msg)

    out = m

    # Center-crop where `m` is larger than target spatial dims
    slices = [slice(None), slice(None)]
    for mi, xi in zip(spatial_m, spatial_x):
        if mi > xi:
            start = (mi - xi) // 2
            end = start + xi
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))
    out = out[tuple(slices)]

    # Symmetric pad where `m` is smaller than target spatial dims
    spatial_cur = out.shape[2:]  # after cropping
    pads = []
    # F.pad expects pads in reverse spatial order:
    # (..., W_left, W_right, H_left, H_right, D_left, D_right)
    for mi, xi in zip(reversed(spatial_cur), reversed(spatial_x)):
        if mi < xi:
            total = xi - mi
            left = total // 2
            right = total - left
        else:
            left = right = 0
        pads.extend([left, right])

    if any(pads):
        # Pad with False (unmasked) == 0.0; applied across all channels
        out = F.pad(out.to(torch.float32), pads, mode="constant", value=0.0) > 0.5

    # Defensive check
    if out.shape != (B_x, C_x, *spatial_x):
        msg = f"Internal error: expected {(B_x, C_x, *spatial_x)}, got {out.shape}."
        logger.error(msg)
        raise RuntimeError(msg)

    return out
