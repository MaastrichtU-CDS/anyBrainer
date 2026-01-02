"""Utility functions for visualizing data."""

__all__ = [
    "as_numpy",
    "pick_volume_slice",
]

from typing import Sequence, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# pyright: reportPrivateImportUsage=false
from monai.transforms import Compose


def as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Torch tensor → numpy; already-numpy objects pass through unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pick_volume_slice(
    vol: np.ndarray,
    slice_index: int,
    axis: int = 0,
    channel: int = 0,
) -> np.ndarray:
    """Extract a single slice from a 3-D or 4-D (C, D, H, W) volume along any
    spatial axis. Falls back gracefully for 2-D data.

    Parameters
    ----------
    vol : np.ndarray
        Input volume with shape (C, D, H, W), (D, H, W), or (H, W)
    slice_index : int
        Index of the slice to extract
    axis : int, optional
        Spatial axis to slice along (0 for D, 1 for H, 2 for W), ignoring the channel dimension
        if present. For example, axis=0 will always slice along D regardless of whether the input
        is (C, D, H, W) or (D, H, W).
    channel : int, optional
        Channel to display if data are four-dimensional (C, D, H, W)
    """
    if vol.ndim == 4:  # (C, D, H, W)  – common for MONAI images
        # First select the channel, then treat remaining (D, H, W) as 3D volume
        return np.take(vol[channel], slice_index, axis=axis)
    elif vol.ndim == 3:  # (D, H, W)     – channel-less volume
        return np.take(vol, slice_index, axis=axis)
    elif vol.ndim == 2:  # already 2-D   – nothing to slice
        return vol
    else:
        raise ValueError(f"Cannot handle volume with shape {vol.shape}")


def visualize_transform_stage(
    pipeline: Compose,
    sample: dict,
    keys: Sequence[str],
    stage: Optional[int] = None,
    slice_indices: Optional[Sequence[int]] = None,
    axis: int = 0,
    cmap: str = "gray",
    channel: int = 0,
    save_path: Optional[str] = None,
):
    """Visualize selected keys from a MONAI dictionary pipeline at an arbitrary
    stage.

    Parameters
    ----------
    pipeline : monai.transforms.Compose
        The composed dictionary transform.
    sample : dict-like
        Input dict (e.g. ``{"img": ..., "brain_mask": ...}``).
    keys : Sequence[str]
        Dictionary keys to show as columns.
    stage : int or None, optional
        *0*  → output after the **first** transform\n
        *1*  → after the **second** transform, …\n
        *None* (default) → after the **full** pipeline.
    slice_indices : Sequence[int] or None, optional
        Axial slice numbers (along depth) to display as rows.
        If *None*, the middle slice is used.
    axis : int, optional
        Spatial axis to display (D, H, W).
    cmap : str, optional
        Matplotlib colour-map for all images (default ``"gray"``).
    channel : int, optional
        Channel to display if data are four-dimensional (C, D, H, W).
    """
    if stage is not None and (stage < 0 or stage >= len(pipeline.transforms)):
        raise IndexError(f"Stage must be in [0, {len(pipeline.transforms)-1}]")

    if axis not in (0, 1, 2):
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    # Run the pipeline up to the requested stage
    current = dict(sample)  # shallow copy – keep original intact
    upto = stage + 1 if stage is not None else len(pipeline.transforms)
    for t in pipeline.transforms[:upto]:
        current = t(current)

    # Assume first key has a 3-D/4-D volume to infer depth
    ref_key = keys[0]
    vol_shape = as_numpy(current[ref_key]).shape
    if len(vol_shape) not in (2, 3, 4):
        raise ValueError(f"Unexpected image dimensionality: {vol_shape}")

    depth = vol_shape[-3 + axis] if len(vol_shape) >= 3 else 1
    if not slice_indices:
        slice_indices = [depth // 2]  # middle slice

    # Prepare the figure grid
    n_rows = len(slice_indices)
    n_cols = len(keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    # Fill in each cell
    for r, idx in enumerate(slice_indices):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            img_np = as_numpy(current[key])
            slice2d = pick_volume_slice(img_np, idx, axis=axis, channel=channel)

            ax.imshow(slice2d, cmap=cmap, vmin=slice2d.min(), vmax=slice2d.max())
            if r == 0:
                ax.set_title(key, fontsize=12)
            ax.set_axis_off()
            if c == 0:
                ax.set_ylabel(f"z={idx}", rotation=0, labelpad=20)

    stage_str = (
        f"stage {stage} (after {stage+1} transform)"
        if stage is not None
        else "final output"
    )
    fig.suptitle(f"MONAI pipeline visualisation – {stage_str}", fontsize=14)
    if save_path is not None:
        fig.savefig(save_path)
    else:

        plt.show()
