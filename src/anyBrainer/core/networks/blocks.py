"""
Pytorch blocks of layers to be used in the construction of networks.
"""

__all__ = [
    'ProjectionHead',
    'ClassificationHead',
    'FusionHead',
    'ConvBNAct3d',
    'FPNLightDecoder3D',
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
            elif x.dim() == 6:  # (B, n_fusion (or n_patches), n_features, D, H, W)
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


class ConvBNAct3d(nn.Module):
    """
    Convolutional block with normalization and activation.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        norm: str = "instance",
        act: str = "SiLU",
        *,
        act_kwargs: dict[str, Any] | None = None,
    ):
        if act_kwargs is None:
            act_kwargs = {}

        super().__init__()
        if norm == "instance":
            Norm = nn.InstanceNorm3d
        elif norm == "group":
            Norm = lambda c: nn.GroupNorm(num_groups=min(8, c), num_channels=c)
        else:
            Norm = nn.BatchNorm3d

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            Norm(out_ch),
            UnitFactory.get_activation_from_kwargs({'name': act, **act_kwargs}),
        )
    def forward(self, x): return self.block(x)


class FPNLightDecoder3D(nn.Module):
    """
    Top-down FPN with narrow width and one 3x3 smooth per level.
    in_chans: channels of encoder features [c1,c2,c3,c4,c5] (f1 highest res)
    width: lateral width (e.g., 32 or 64)
    """
    def __init__(
        self,
        in_feats: Sequence[int],
        in_chans: int,
        out_channels: int,
        width: int = 32,
        norm: str = "instance"
    ):
        super().__init__()
        assert len(in_feats) == 5, "Need 5 scales [f1..f5]"
        f1,f2,f3,f4,f5 = in_feats

        self.in_stem = ConvBNAct3d(in_chans, width, k=3, p=1, norm=norm)

        self.lat1 = nn.Conv3d(f1, width, kernel_size=1, bias=False)
        self.lat2 = nn.Conv3d(f2, width, kernel_size=1, bias=False)
        self.lat3 = nn.Conv3d(f3, width, kernel_size=1, bias=False)
        self.lat4 = nn.Conv3d(f4, width, kernel_size=1, bias=False)
        self.lat5 = nn.Conv3d(f5, width, kernel_size=1, bias=False)

        # one light smooth per pyramid level after lateral+topdown add
        self.smooth4 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth3 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth2 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth1 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth0 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)

        # head on the finest map (keeps params tiny)
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [f1,f2,f3,f4,f5] high->low res
        if len(feats) != 5:
            msg = (f"[{self.__class__.__name__}] Expected 5 features, "
                   f"got {len(feats)}.")
            logger.error(msg)
            raise ValueError(msg)
        
        f1,f2,f3,f4,f5 = feats
        p5 = self.lat5(f5) # bottleneck
        p4 = self.smooth4(self.lat4(f4) + self._upsample_to(p5, f4))
        p3 = self.smooth3(self.lat3(f3) + self._upsample_to(p4, f3))
        p2 = self.smooth2(self.lat2(f2) + self._upsample_to(p3, f2))
        p1 = self.smooth1(self.lat1(f1) + self._upsample_to(p2, f1))
        p0 = self.smooth0(self.in_stem(x) + self._upsample_to(p1, x))
        return self.head(p0) # logits at full res