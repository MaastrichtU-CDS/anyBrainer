"""Utility functions for inferers."""

__all__ = [
    "get_patch_gaussian_weight",
]

from typing import Any, Sequence
import math

import torch

def ensure_tuple_of_length(
    value: Any | Sequence[Any],
    length: int,
) -> tuple[Any, ...]:
    """Ensure that a value is a tuple of length `length`."""
    if isinstance(value, (list, tuple)):
        if len(value) == length:
            return tuple(value)
        else:
            msg = f"Expected tuple of length {length}, got {len(value)}"
            raise ValueError(msg)
    else:
        return tuple([value] * length)

def get_patch_gaussian_weight(
    patch_center: Sequence[int | float],
    image_center: Sequence[int | float],
    sigma: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Gaussian weight exp(-||c-μ||² / 2σ²).
    """
    sq_dist = sum(((c - m) / s) ** 2 for c, m, s in zip(patch_center, image_center, sigma))
    return math.exp(-0.5 * sq_dist)
