"""
Pytorch blocks of layers to be used in the construction of networks.
"""

__all__ = [
    'ProjectionHead',
]

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Two‑layer MLP with L2‑normalised output."""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )
        logger.info(f"ProjectionHead initialized with in_dim={in_dim}, "
                    f"hidden_dim={hidden_dim}, proj_dim={proj_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalisation
        return x
