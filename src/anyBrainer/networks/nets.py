"""
Pytorch networks by combining blocks.
"""

__all__ = [
    'Swinv2CL',
]

import logging
from typing import Sequence

import torch
import torch.nn as nn
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
)

logger = logging.getLogger(__name__)


class Swinv2CL(nn.Module):
    """Swin ViT V2 for Contrastive Learning."""
    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict = {},
        proj_dim: int = 128,
        proj_hidden_dim: int = 2048,
        proj_hidden_act: str = "gelu",
        aux_mlp_head: bool = True,
        aux_mlp_num_classes: int = 7,
        aux_mlp_hidden_dim: int | None = None,
        aux_mlp_hidden_act: str = "gelu",
        aux_mlp_dropout: float = 0.0,
    ):
        super().__init__()

        # SwinViT encoder
        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)

        self.encoder = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths, 
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
            use_v2=use_v2,
            **extra_swin_kwargs,
        )
        logger.info(f"[Swinv2CL] Encoder initialized with in_channels={in_channels}, "
                    f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
                    f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
                    f"extra_swin_kwargs={extra_swin_kwargs}")

        # Projection head
        self.projection_head = ProjectionHead(
            in_dim=feature_size * 16,  # C*2*2*2*2
            hidden_dim=proj_hidden_dim,
            proj_dim=proj_dim,
            activation=proj_hidden_act,
            model_name="Swinv2CL",
        )

        # Optional auxiliary classification head
        if aux_mlp_head:
            self.classification_head = ClassificationHead(
                in_dim=feature_size * 16,
                num_classes=aux_mlp_num_classes,
                hidden_dim=aux_mlp_hidden_dim,
                activation=aux_mlp_hidden_act,
                dropout=aux_mlp_dropout,
                model_name="Swinv2CL",
            )
        else:
            self.classification_head = None
            logger.info("[Swinv2CL] Skipping auxiliary classification head (aux_mlp_head=False)")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.encoder(x)
        proj = self.projection_head(x[-1]) # last layer

        if self.classification_head is not None:
            aux = self.classification_head(x[-1])
        else:
            aux = None

        return proj, aux