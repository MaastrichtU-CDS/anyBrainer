"""
Pytorch blocks of layers to be used in the construction of networks.
"""

__all__ = [
    'ProjectionHead',
    'ClassificationHead',
]

import logging
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anyBrainer.core.networks.utils import get_mlp_head_args
from anyBrainer.factories.unit import UnitFactory

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Two‑layer MLP with L2‑normalised output."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        proj_dim: int = 128,
        activation: str = "GELU",
        *,
        activation_kwargs: dict[str, Any] | None = None,
        model_name: str | None = None,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if activation_kwargs is None:
            activation_kwargs = {}

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            UnitFactory.get_activation_from_kwargs(
                {'name': activation, **activation_kwargs}
            ),
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
            msg = f"Unexpected input shape {x.shape}"
            logger.error(msg)
            raise ValueError(msg)
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalisation
        return x


class ClassificationHead(nn.Module):
    """Optional dropout + MLP classifier with global average pooling."""
    def __init__(
        self, 
        in_dim: int,
        num_classes: int,
        num_hidden_layers: int = 0,
        dropout: Sequence[float] | float = 0.0,
        hidden_dim: Sequence[int] | int = [],
        activation: Sequence[str] | str = "GELU",
        *,
        activation_kwargs: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        model_name: str | None = None,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if activation_kwargs is None:
            activation_kwargs = {}

        dropouts, hidden_dims, activations, activation_kwargs = get_mlp_head_args(
            in_dim, num_classes, num_hidden_layers, dropout, hidden_dim, activation, activation_kwargs
        )

        layers = []
        for i in range(num_hidden_layers + 1):
            layers.extend([
                nn.Dropout(p=dropouts[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                UnitFactory.get_activation_from_kwargs(
                    {'name': activations[i], **activation_kwargs[i]}
                ),
            ])

        self.classifier = nn.Sequential(*layers)

        # Log hyperparameters
        summary = (
            f"in_dim={in_dim}, num_classes={num_classes}, "
            f"num_hidden_layers={num_hidden_layers}, dropout(s)={dropouts}, "
            f"hidden_dim(s)={hidden_dims}, activation(s)={activations}"
        )
        logger.info(f"[{model_name}] ClassificationHead initialized with {summary}"
                    if model_name else f"ClassificationHead initialized with {summary}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = self.global_pool(x).view(x.size(0), -1)  # (B, C)
        elif x.ndim == 2:
            pass  # Already flat
        else:
            msg = f"Unexpected input shape {x.shape}"
            logger.error(msg)
            raise ValueError(msg)
        return self.classifier(x)
