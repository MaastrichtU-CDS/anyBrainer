"""Visualization utilities.

Displays volume slices in three orthogonal planes.
Volume shape: (C, D1, D2, D3) where C is channels and D1, D2, D3 are spatial dimensions.
"""

from __future__ import annotations

__all__ = [
    "plot_npy_volumes",
    "to_uint8_image",
]

import logging

from numpy.typing import NDArray
import torch

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _get_mid_slices(volume: NDArray) -> tuple[int, int, int]:
    """Get mid slice indices for each spatial dimension.

    Args:
        volume: 4D volume (C, D1, D2, D3) or 3D volume (D1, D2, D3)

    Returns:
        Mid slice indices for (dim1, dim2, dim3)
    """
    if volume.ndim == 4:
        C, D1, D2, D3 = volume.shape
    elif volume.ndim == 3:
        D1, D2, D3 = volume.shape
        C = 1
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")

    # Mid slices for each dimension
    mid_d1 = D1 // 2
    mid_d2 = D2 // 2
    mid_d3 = D3 // 2

    return mid_d1, mid_d2, mid_d3


def plot_npy_volumes(
    volumes: dict[str, NDArray],
    slice_indices: tuple[int, int, int] | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (18, 12),
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
) -> None:
    """Plot multiple volumes in a single figure for comparison.

    Args:
        volumes: Dict mapping volume names to volume arrays (C, D1, D2, D3)
        slice_indices: Custom slice indices (dim1, dim2, dim3)
        titles: Column titles for each plane (default: ["Dim2-Dim3", "Dim1-Dim3", "Dim1-Dim2"])
        figsize: Figure size
        cmap: Colormap
        vmin, vmax: Value range for colormap
        save_path: Path to save figure (optional)
    """
    volume_names = list(volumes.keys())
    n_volumes = len(volume_names)

    # Default titles for the three orthogonal planes
    # Note: imshow displays first array dim as y-axis, second as x-axis
    sub_titles = ["Dim3-Dim2", "Dim3-Dim1", "Dim2-Dim1"]
    main_title = title or "Multi-Volume Comparison - Three Orthogonal Planes"

    # Create figure with subplots (n_volumes rows, 3 columns)
    fig, axes = plt.subplots(n_volumes, 3, figsize=figsize)

    # Handle single volume case
    if n_volumes == 1:
        axes = axes.reshape(1, -1)

    # Get slice indices (use first volume to determine dimensions)
    first_volume = volumes[volume_names[0]]
    if slice_indices is None:
        slice_indices = _get_mid_slices(first_volume)

    idx_d1, idx_d2, idx_d3 = slice_indices

    # Plot each volume
    for row_idx, vol_name in enumerate(volume_names):
        volume = volumes[vol_name]

        # Handle channel dimension - extract (D1, D2, D3) from (C, D1, D2, D3)
        if volume.ndim == 4:
            volume = volume[0]

        # Plane 1: Slice through dim1, shows dim2-dim3 plane
        # volume[idx_d1, :, :] gives us (D2, D3)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im1 = axes[row_idx, 0].imshow(
            volume[idx_d1, :, :],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row_idx, 0].set_title(
            f"{vol_name}\n{sub_titles[0]} (D1={idx_d1})", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 0].set_xlabel("Dim3")
        axes[row_idx, 0].set_ylabel("Dim2")
        axes[row_idx, 0].grid(False)
        plt.colorbar(im1, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Plane 2: Slice through dim2, shows dim1-dim3 plane
        # volume[:, idx_d2, :] gives us (D1, D3)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im2 = axes[row_idx, 1].imshow(
            volume[:, idx_d2, :],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row_idx, 1].set_title(
            f"{vol_name}\n{sub_titles[1]} (D2={idx_d2})", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 1].set_xlabel("Dim3")
        axes[row_idx, 1].set_ylabel("Dim1")
        axes[row_idx, 1].grid(False)
        plt.colorbar(im2, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Plane 3: Slice through dim3, shows dim1-dim2 plane
        # volume[:, :, idx_d3] gives us (D1, D2)
        # imshow: first array dim -> y-axis, second array dim -> x-axis
        im3 = axes[row_idx, 2].imshow(
            volume[:, :, idx_d3],
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row_idx, 2].set_title(
            f"{vol_name}\n{sub_titles[2]} (D3={idx_d3})", fontsize=10, fontweight="bold"
        )
        axes[row_idx, 2].set_xlabel("Dim2")
        axes[row_idx, 2].set_ylabel("Dim1")
        axes[row_idx, 2].grid(False)
        plt.colorbar(im3, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

    plt.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def to_uint8_image(
    img2d: torch.Tensor,
    *,
    clamp_val: tuple[float, float] | None = None,
    clamp_perc: tuple[float, float] | None = None,
    scale_to: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Convert a 2D tensor to uint8 image (0..255) for logging (e.g.
    wandb.Image).

    Args:
        img2d: (H, W) tensor (any dtype/device).
        clamp_val: Optional (lo, hi) clamp applied before scaling.
               Example for z-norm: (-3.0, 3.0)
        clamp_perc: Optional (lo, hi) percentile clamp applied before scaling.
               Example for 0.5th and 99.5th percentiles: (0.5, 99.5)
        scale_to: Optional (lo, hi) defining the intensity window mapped to [0,255].
               If provided, values <= lo map to 0, >= hi map to 255.
               If not provided:
                 - uses `clamp` as the window if clamp is set
                 - else uses tensor min/max

    Returns:
        (H, W) torch.uint8 on CPU.
    """
    if img2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got shape={tuple(img2d.shape)}")

    x = img2d.detach().to(dtype=torch.float32, device="cpu")
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if clamp_val is not None:
        c_lo, c_hi = float(clamp_val[0]), float(clamp_val[1])
        if c_hi <= c_lo:
            msg = f"Invalid clamp range: {clamp_val}, lo={c_lo} should be less than hi={c_hi}"
            logger.error(msg)
            raise ValueError(msg)
        x = x.clamp(c_lo, c_hi)
    elif clamp_perc is not None:
        c_lo, c_hi = float(clamp_perc[0]), float(clamp_perc[1])
        if c_hi <= c_lo:
            msg = f"Invalid clamp range: {clamp_perc}, lo={c_lo} should be less than hi={c_hi}"
            logger.error(msg)
            raise ValueError(msg)
        lo = torch.quantile(x, c_lo / 100.0).item()
        hi = torch.quantile(x, c_hi / 100.0).item()
        x = x.clamp(lo, hi)

    if scale_to is not None:
        lo, hi = float(scale_to[0]), float(scale_to[1])
        if hi <= lo:
            msg = (
                f"Invalid scale range: {scale_to}, lo={lo} should be less than hi={hi}"
            )
            logger.error(msg)
            raise ValueError(msg)
    else:
        lo = float(x.min().item())
        hi = float(x.max().item())
        if hi <= lo + 1e-8:
            return torch.zeros_like(x, dtype=torch.uint8)

    # window -> [0,1] -> [0,255]
    x = (x - lo) / (hi - lo + 1e-8)
    x = (x * 255.0).clamp(0.0, 255.0)
    return x.to(torch.uint8)
