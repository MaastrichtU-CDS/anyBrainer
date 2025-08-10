"""
Pytorch blocks of layers to be used in the construction of networks.
"""

__all__ = [
    'ProjectionHead',
    'ClassificationHead',
    'FusionHead',
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
        msg = (f"[{self.__class__.__name__}] ProjectionHead initialized with in_dim={in_dim}, "
               f"hidden_dim={hidden_dim}, proj_dim={proj_dim}")
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
            f"linear_layer(s)={hidden_dims}, activation(s)={activations}"
        )
        logger.info(f"[{self.__class__.__name__}] ClassificationHead initialized with {summary}")

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


class FusionHead(nn.Module):
    """
    Unify representation across brain patches and different modalities.

    Designed for low-data regimes; the only learnable parameters
    are the modality fusion weights; rest is global average pooling.

    Set `mod_only=True` to only fuse modalities, and `mod_only=False` to perform
    global average pooling on spatial features and n_patches.

    Input shape: (B, n_fusion, n_patches, n_features, D, H, W);
        typically output by an encoder.
    Output shape: (B, n_features)
    """
    def __init__(
        self, 
        n_fusion: int,
        mod_only: bool = False,
    ):
        super().__init__()
        self.n_fusion = n_fusion 
        self.mod_only = mod_only
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion))

        logger.info(f"[{self.__class__.__name__}] FusionHead initialized with "
                    f"n_fusion={n_fusion}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mod_only: # Fuse everything
            if x.dim() == 7: # (B, n_fusion, n_patches, n_features, D, H, W)
                pass
            elif x.dim() == 6: # (B, n_patches, n_features, D, H, W)
                x = x.unsqueeze(1)
            elif x.dim() == 5: # (B, n_features, D, H, W)
                x = x.unsqueeze(1).unsqueeze(2)
            else:
                msg = f"[{self.__class__.__name__}] Unexpected input shape {x.shape}"
                logger.error(msg)
                raise ValueError(msg)
            x = x.mean(dim=[-3, -2, -1]) # (B, n_fusion, n_patches, n_features)
            x = x.mean(dim=2) # (B, n_fusion, n_features)
        else: # Already fused, except modalities
            if x.dim() != 3:
                msg = (f"[{self.__class__.__name__}] Invalid input shape {x.shape} "
                       f"for `mod_only` mode; expected (B, n_fusion, n_features).")
                logger.error(msg)
                raise ValueError(msg)
            
        if x.size(1) != self.n_fusion:
            msg = (f"[{self.__class__.__name__}] Unexpected number of fusion channels; "
                f"expected {self.n_fusion}, got {x.size(1)}")
            logger.error(msg)
            raise ValueError(msg)
            
        weights = torch.softmax(self.fusion_weights, dim=0)  # (n_fusion,)
        x = (x * weights.view(1, -1, 1)).sum(dim=1) # (B, n_features)
        return x