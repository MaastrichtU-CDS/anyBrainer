"""Utility functions for creating and running models."""

import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")