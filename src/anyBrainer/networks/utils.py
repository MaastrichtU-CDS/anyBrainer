"""Utility functions for creating and running models."""

__all__ = [
    "get_act_fn",
]

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def get_act_fn(name: str) -> nn.Module:
    """Get activation function by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    else:
        msg = f"Unsupported activation: {name}"
        logger.error(msg)
        raise ValueError(msg)