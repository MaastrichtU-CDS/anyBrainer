"""
Pytorch networks by combining blocks.
"""

__all__ = [
    'Swinv2CL',
    'Swinv2Classifier',
    'Swinv2ClassifierMidFusion',
]

import logging
from typing import Sequence, Any, Literal, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.core.utils import (
    ensure_tuple_dim,
    noisy_or_bag_logits,
    lse_bag_logits_from_instance_logits,
    topk_mean_bag_logits,
)
from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
    FusionHead,
)

logger = logging.getLogger(__name__)


@register(RK.NETWORK)
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
        extra_swin_kwargs: dict[str, Any] | None = None,
        spatial_dims: int = 3,
        proj_dim: int = 128,
        proj_hidden_dim: int = 2048,
        proj_hidden_act: str = "GELU",
        aux_mlp_head: bool = True,
        aux_mlp_num_classes: int = 5,
        aux_mlp_num_hidden_layers: int = 0,
        aux_mlp_hidden_dim: int = 0,
        aux_mlp_hidden_act: str = "GELU",
        aux_mlp_dropout: float = 0.0,
    ):
        super().__init__()

        # SwinViT encoder
        if extra_swin_kwargs is None:
            extra_swin_kwargs = {}
        
        self.encoder = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths, 
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
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
        )

        # Optional auxiliary classification head
        if aux_mlp_head:
            self.classification_head = ClassificationHead(
                in_dim=feature_size * 16,
                num_hidden_layers=aux_mlp_num_hidden_layers,
                num_classes=aux_mlp_num_classes,
                hidden_dim=aux_mlp_hidden_dim,
                activation=aux_mlp_hidden_act,
                dropout=aux_mlp_dropout,
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


@register(RK.NETWORK)
class Swinv2Classifier(nn.Module):
    """
    Swin ViT V2 for image-level classification.
    
    Supports late-multimodal fusion. 
    
    - If late_fusion is True, the input should be a tensor with shape
        (B, n_late_fusion, n_patches, C, *spatial_dims).
        
        The input is then permuted into (n_late_fusion, n_patches, B, C, *spatial_dims)
        and fed into the encoder in chunks of size B for n_late_fusion * n_patches times.
        
        Combined encoder outputs are permuted to (B, n_late_fusion, n_patches, *feats)
        and fed into the fusion head, which produces a flat 2D tensor (B, in_dim) for subsequent 
        processing by the classification head.
        
    - If late_fusion is False, the input should be a 4D tensor (B, C, *patch_size) and
        passed through the encoder only once.

    The return value is a 2D tensor with the raw logits (B, num_classes).

    """
    def __init__(
        self,
        *,
        # SwinViT encoder args
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict[str, Any] | None = None,
        spatial_dims: int = 3,

        # MLP classification head args
        mlp_num_classes: int = 2,
        mlp_num_hidden_layers: int = 1,
        mlp_hidden_dim: int | Sequence[int] = 384,
        mlp_dropout: float | Sequence[float] = 0.3,
        mlp_activations: str | Sequence[str] = "GELU",
        mlp_activation_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,

        # Aggregate inputs
        late_fusion: bool = False,
        n_late_fusion: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.late_fusion = late_fusion
        self.n_late_fusion = n_late_fusion

        # SwinViT encoder
        if extra_swin_kwargs is None:
            extra_swin_kwargs = {}
        
        self.encoder = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths, 
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
            use_v2=use_v2,
            **extra_swin_kwargs,
        )
        logger.info(f"[Swinv2Classifier] Encoder initialized with in_channels={in_channels}, "
                    f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
                    f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
                    f"extra_swin_kwargs={extra_swin_kwargs}")
        
        self.classification_head = ClassificationHead(
            in_dim=feature_size * 16,
            num_hidden_layers=mlp_num_hidden_layers,
            num_classes=mlp_num_classes,
            hidden_dim=mlp_hidden_dim,
            activation=mlp_activations,
            dropout=mlp_dropout,
        )

        if self.late_fusion:
            self.fusion_head = FusionHead(
                n_fusion=self.n_late_fusion,
                mod_only=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_late_fusion, n_patches, C, *spatial_dims).
        
        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        if self.late_fusion:
            if x.ndim != 4 + self.spatial_dims:
                msg = (f"[Swinv2Classifier] late_fusion=True expects input with "
                       f"{4 + self.spatial_dims} dims "
                       f"(B, n_late_fusion, n_patches, C, *spatial_dims); "
                       f"got {tuple(x.shape)}")
                logger.error(msg)
                raise ValueError(msg)
            
            B, n_late_fusion, n_patches, C, *spatial = x.shape

            if n_late_fusion != self.n_late_fusion:
                msg = (f"[Swinv2Classifier] Expected n_late_fusion={self.n_late_fusion}, "
                       f"got {n_late_fusion}.")
                logger.error(msg)
                raise ValueError(msg)
            
            if C != self.in_channels:
                msg = (f"[Swinv2Classifier] in_channels={self.in_channels} but "
                       f"input has {C} channels.")
                logger.error(msg)
                raise ValueError(msg)
            
            if len(spatial) != self.spatial_dims:
                msg = (f"[Swinv2Classifier] spatial_dims={self.spatial_dims} but "
                       f"input has {len(spatial)} spatial dimensions.")
                logger.error(msg)
                raise ValueError(msg)
            
            # Split input into chunks of size B and pass to encoder
            mod_vecs = []
            for m in range(n_late_fusion):
                vec_sum = None
                for p in range(n_patches): # stream over patches to avoid keeping many feature maps
                    xmp = x[:, m, p].contiguous() # (B, C, *spatial)
                    f_map = self.encoder(xmp)[-1] # (B, n_feats, *bottleneck_dims)  last-scale feature
                    v = f_map.mean(dim=tuple(range(2, f_map.ndim))) # GAP -> (B, n_feats)
                    vec_sum = v if vec_sum is None else vec_sum + v
                mod_vecs.append(cast(torch.Tensor, vec_sum) / n_patches) # (B, n_feats)

            feats = torch.stack(mod_vecs, dim=1) # (B, n_fusion, n_feats)
            feats = self.fusion_head(feats) # (B, in_dim)
        else:
            f_map = self.encoder(x)[-1] # (B, n_feats, *bottleneck_dims)
            feats = f_map.mean(dim=tuple(range(2, f_map.ndim))) # GAP -> (B, n_feats)

        return self.classification_head(feats)


@register(RK.NETWORK)
class Swinv2ClassifierMidFusion(nn.Module):
    """
    Swin ViT V2 classifier with mid-level fusion.

    Contrary to `Swinv2Classifier`, this model preserves the spatial dimensions
    of the feature maps, performs multimodal fusion, and then aggregates them to
    a single patch-level prediction using a bagging approach defined by the 
    `aggregator` argument.

    The return value is a 2D tensor with the raw logits (B, num_classes).
    """
    def __init__(self,
        *,
        # SwinViT encoder args
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict[str, Any] | None = None,

        # Head / fusion
        n_classes: int = 1,
        n_fusion: int = 1,
        spatial_dims: int = 3,
        share_encoder: bool = True,
        aggregator: Literal["noisy_or", "lse", "topk"] = "noisy_or",
        tau: float = 2.0,
        topk: int | None = None,
    ):  
        """
        Args:
            n_classes: Number of output classes.
            n_fusion: Number of modalities to fuse.
            share_encoder: Whether to share the encoder weights across modalities.
            aggregator: Aggregation method.
            tau: Temperature for LSE / smoothing.
            topk: Number of top-k cells to aggregate.

        See MONAI's `SwinViT` (inside `monai.networks.nets.swin_unetr`) for
        encoder args.
        """
        super().__init__()

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.n_classes = n_classes
        self.n_fusion = n_fusion
        self.share_encoder = share_encoder
        self.aggregator = aggregator
        self.tau = tau
        self.topk = topk

        # SwinViT encoder
        if extra_swin_kwargs is None:
            extra_swin_kwargs = {}
        
        # Build encoders
        def _make_encoder():
            return SwinViT(
                in_chans=in_channels,
                embed_dim=feature_size,
                depths=depths,
                num_heads=num_heads,
                window_size=ensure_tuple_dim(window_size, spatial_dims),
                patch_size=ensure_tuple_dim(patch_size, spatial_dims),
                use_v2=use_v2,
                **extra_swin_kwargs,
            )
        
        if share_encoder:
            self.encoders = nn.ModuleList([_make_encoder()])
            self._share_encoder = True
        else:
            self.encoders = nn.ModuleList([_make_encoder() for _ in range(n_fusion)])
            self._share_encoder = False
        
        # 1x1x1 scoring head: (B, n_feats, *spatial_dims) -> (B, n_classes, *spatial_dims)
        self.head = nn.Conv3d(feature_size * 16, n_classes, 1)

        # Mid-level fusion weights across modalities (applied on logits)
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion))

        logger.info(f"[Swinv2ClassifierMidFusion] Encoder initialized with in_channels={in_channels}, "
                    f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
                    f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
                    f"extra_swin_kwargs={extra_swin_kwargs}. Heads initialized with in_dim={feature_size * 16}, "
                    f"n_classes={n_classes}, n_fusion={n_fusion}.")
    
    def _encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and extract the last feature map (B, F, d, h, w)."""
        f_maps = encoder(x)
        if isinstance(f_maps, (tuple, list)):
            f_map = f_maps[-1]
        return f_map # (B, n_feats, *spatial_dims)
    
    def _bag_logits(self, cell_logits: torch.Tensor) -> torch.Tensor:
        """
        cell_logits: (B, n_classes, *spatial_dims)  ->  bag logits: (B, n_classes)
        """
        dims = tuple(range(-self.spatial_dims, 0))  # (-3,-2,-1) for 3D
        if self.aggregator == "noisy_or":
            return noisy_or_bag_logits(cell_logits, dims)
        elif self.aggregator == "lse":
            return lse_bag_logits_from_instance_logits(cell_logits, dims, tau=self.tau)
        elif self.aggregator == "topk":
            k = self.topk if self.topk is not None else 8
            return topk_mean_bag_logits(cell_logits, dims, k=k)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input from (B, n_late_fusion, *spatial_dims) to 
        (B, n_fusion, C, *spatial_dims).
        """
        if x.ndim == self.spatial_dims + 2: # n_late_fusion in channel_dim
            x = x.unsqueeze(2)
        elif x.ndim == self.spatial_dims + 3: # channel_dim already in place
            pass
        else:
            msg = (f"[{self.__class__.__name__}] Expected input shape to be "
                    f"(B, N, *spatial_dims) or (B, N, C, *spatial_dims), "
                    f"but got {x.shape}.")
            logger.error(msg)
            raise ValueError(msg)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_fusion, C, *spatial_dims).
        
        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        x = self._reshape_input(x)
        
        if x.ndim != 3 + self.spatial_dims:
            msg = (f"[Swinv2ClassifierMidFusion] Expected input with "
                   f"{3 + self.spatial_dims} dims "
                   f"(B, n_fusion, C, *spatial_dims); "
                   f"got {tuple(x.shape)}")
            logger.error(msg)
            raise ValueError(msg)
        
        B, n_fusion, C, *spatial = x.shape

        if n_fusion != self.n_fusion:
            msg = (f"[Swinv2ClassifierMidFusion] Expected n_fusion={self.n_fusion}, "
                   f"got {n_fusion}.")
            logger.error(msg)
            raise ValueError(msg)
        
        if C != self.in_channels:
            msg = (f"[Swinv2ClassifierMidFusion] in_channels={self.in_channels} but "
                   f"input has {C} channels.")
            logger.error(msg)
            raise ValueError(msg)

        if len(spatial) != self.spatial_dims:
            msg = (f"[Swinv2ClassifierMidFusion] spatial_dims={self.spatial_dims} but "
                   f"input has {len(spatial)} spatial dimensions.")
            logger.error(msg)
            raise ValueError(msg)
        
        # Get cell-level scores per modality
        per_mod_logits = []
        for m in range(n_fusion):
            encoder = self.encoders[0] if self._share_encoder else self.encoders[m]
            feats = self._encode(encoder, x[:, m]) # (B, n_feats, *spatial_dims)
            logits = self.head(feats) # (B, n_classes, *spatial_dims)
            per_mod_logits.append(logits)
        
        # Mid-level fusion on logits (weighted sum across modalities)
        w = torch.softmax(self.fusion_weights, dim=0) # (n_fusion,)
        fused = torch.stack(per_mod_logits, dim=0) # (n_fusion, B, n_classes, *spatial_dims)
        fused = (w[:, None, None, None, None, None] * fused).sum(dim=0)  # (B, n_classes, *spatial_dims)

        # MIL pooling to bag logits
        return self._bag_logits(fused) # (B, n_classes)