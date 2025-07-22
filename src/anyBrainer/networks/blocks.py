"""
Pytorch blocks of layers to be used in the construction of networks.
"""

__all__ = [
    'ProjectionHead',
    'ClassificationHead',
]

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from anyBrainer.networks.utils import get_act_fn

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Two‑layer MLP with L2‑normalised output."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        proj_dim: int = 128,
        activation: str = "gelu",
        *,
        model_name: str | None = None,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            get_act_fn(activation),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )

        # Log hyperparameters
        msg = (f"ProjectionHead initialized with in_dim={in_dim}, "
               f"hidden_dim={hidden_dim}, proj_dim={proj_dim}")
        if model_name is not None:
            msg = f"[{model_name}] " + msg
        logger.info(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:  # Apply pooling if input is 3D
            x = self.global_pool(x).view(x.size(0), -1)
        elif x.ndim == 2:
            pass  # Already flat
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalisation
        return x


class ClassificationHead(nn.Module):
    """Optional dropout + MLP classifier with global average pooling."""
    def __init__(
        self, in_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
        activation: str = "gelu",
        *,
        model_name: str | None = None,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_dim, hidden_dim),
                get_act_fn(activation),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_dim, num_classes)
            )
        
        # Log hyperparameters
        msg = (f"ClassificationHead initialized with in_dim={in_dim}, "
               f"num_classes={num_classes}, dropout={dropout}, "
               f"hidden_dim={hidden_dim}, activation={activation}")
        if model_name is not None:
            msg = f"[{model_name}] " + msg
        logger.info(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = self.global_pool(x).view(x.size(0), -1)  # (B, C)
        elif x.ndim == 2:
            pass  # Already flat
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")
        return self.classifier(x)
