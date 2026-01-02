"""Utility functions for inferers."""

__all__ = [
    "get_patch_gaussian_weight",
]

from typing import Sequence
import math


def get_patch_gaussian_weight(
    patch_center: Sequence[int | float],
    image_center: Sequence[int | float],
    sigma: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Gaussian weight exp(-||c-μ||² / 2σ²)."""
    sq_dist = sum(
        ((c - m) / s) ** 2 for c, m, s in zip(patch_center, image_center, sigma)
    )
    return math.exp(-0.5 * sq_dist)
