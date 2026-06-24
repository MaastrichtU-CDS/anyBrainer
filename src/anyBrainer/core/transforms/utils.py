"""Utility functions for transforms."""

__all__ = [
    "SegImagePadMode",
    "assign_key",
    "resolve_seg_image_pad_modes",
    "scale_spatial_size",
]

import logging
from typing import Literal, Sequence

logger = logging.getLogger(__name__)

SegImagePadMode = Literal["zeros", "border"]


def assign_key(data, key):
    try:
        return data[key]
    except Exception as e:
        msg = f"key {key} not found in data"
        logger.error(msg)
        raise ValueError(msg) from e


def resolve_seg_image_pad_modes(pad_img: SegImagePadMode) -> tuple[str, str]:
    """Map user-facing image padding to MONAI affine and spatial modes."""
    modes: dict[str, tuple[str, str]] = {
        "zeros": ("zeros", "constant"),
        "border": ("border", "edge"),
    }
    try:
        return modes[pad_img]
    except KeyError as e:
        msg = f"pad_img must be one of {sorted(modes)}, got {pad_img!r}."
        logger.error(msg)
        raise ValueError(msg) from e


def scale_spatial_size(
    spatial_size: int | Sequence[int],
    factor: float,
) -> tuple[int, int, int]:
    """Scale the last three spatial dims by ``factor`` (rounded, minimum 1)."""
    if factor <= 0:
        msg = f"pre_spatial_scale_factor must be positive, got {factor}."
        logger.error(msg)
        raise ValueError(msg)
    if isinstance(spatial_size, int):
        dims = (spatial_size, spatial_size, spatial_size)
    else:
        dims = tuple(spatial_size)
        if len(dims) != 3:
            msg = (
                f"input_size must be an int or length-3 sequence, got {spatial_size!r}."
            )
            logger.error(msg)
            raise ValueError(msg)
    return (
        max(1, int(round(dims[0] * factor))),
        max(1, int(round(dims[1] * factor))),
        max(1, int(round(dims[2] * factor))),
    )
