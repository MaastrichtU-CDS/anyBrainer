"""Pytorch networks by combining blocks."""

__all__ = [
    "Swinv2CL",
    "Swinv2Classifier",
    "Swinv2ClassifierMidFusion",
    "Swinv2LateFusionFPNDecoder",
    "SwinMIM",
    "Multimodal3DSwinMIMFPN",
]

import logging
from typing import Sequence, Any, Literal, cast

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.nets.dynunet import DynUNet
from monai.networks.blocks.unetr_block import UnetrBasicBlock as MONAIUnetrBasicBlock

from anyBrainer.core.utils import (
    ensure_tuple_dim,
    noisy_or_bag_logits,
    lse_bag_logits_from_instance_logits,
    topk_mean_bag_logits,
)
from anyBrainer.core.networks.utils import merge_mask_nd
from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
    FusionHead,
    FPNLightDecoder3D,
    FPNDecoder3DFeaturesOnly,
    VoxelShuffleHead3D,
    MaskToken,
    PartialUnetrBasicBlock,
    MultimodalPatchEmbed,
    MultimodalPatchEmbedAdapter,
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
        logger.info(
            f"[Swinv2CL] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}"
        )

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
            logger.info(
                "[Swinv2CL] Skipping auxiliary classification head (aux_mlp_head=False)"
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.encoder(x)
        proj = self.projection_head(x[-1])  # last layer

        if self.classification_head is not None:
            aux = self.classification_head(x[-1])
        else:
            aux = None

        return proj, aux


@register(RK.NETWORK)
class Swinv2Classifier(nn.Module):
    """Swin ViT V2 for image-level classification.

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
        expect_patch_dim: bool = True,
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
        logger.info(
            f"[Swinv2Classifier] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}"
        )

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
        self.expect_patch_dim = expect_patch_dim

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to required shape for late fusion.

        Args:
            x: (B, n_late_fusion, [n_patches, ]C, *spatial_dims) or
                (B, [n_patches, ]n_late_fusion, *spatial_dims)

        Returns:
            torch.Tensor: (B, n_late_fusion, n_patches, C, *spatial_dims)
        """
        # Create n_patches dim if it doesn't exist
        if x.ndim == 2 + int(self.expect_patch_dim) + self.spatial_dims:
            # (B, [n_patches, ]n_late_fusion, *spatial_dims)
            if not self.expect_patch_dim:  # (B, n_late_fusion, *spatial_dims)
                x = x.unsqueeze(1)  # (B, 1, n_late_fusion, *spatial_dims)
            x = x.unsqueeze(3)  # (B, n_patches, n_late_fusion, 1, *spatial_dims)
            x = x.permute(
                0, 2, 1, 3, *range(4, x.ndim)
            )  # (B, n_late_fusion, n_patches, 1, *spatial_dims)
        elif x.ndim == 3 + int(self.expect_patch_dim) + self.spatial_dims:
            # (B, n_late_fusion, [n_patches, ]C, *spatial_dims)
            if not self.expect_patch_dim:  # (B, n_late_fusion, C, *spatial_dims)
                x = x.unsqueeze(2)  # (B, n_late_fusion, 1, C, *spatial_dims)
        else:
            msg = (
                f"[{self.__class__.__name__}] Expected input shape to be "
                f"(B, [n_patches, ]n_late_fusion, *spatial_dims) or "
                f"(B, n_late_fusion, [n_patches, ]C, *spatial_dims), "
                f"but got {x.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_mods, [n_patches, ], C, *spatial_dims) or
               (B, [n_patches, ]n_mods, *spatial_dims).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        if self.late_fusion:
            x = self._reshape_input(x)
            B, n_late_fusion, n_patches, C, *spatial = x.shape

            if n_late_fusion != self.n_late_fusion:
                msg = (
                    f"[Swinv2Classifier] Expected n_late_fusion={self.n_late_fusion}, "
                    f"got {n_late_fusion}."
                )
                logger.error(msg)
                raise ValueError(msg)

            if C != self.in_channels:
                msg = (
                    f"[Swinv2Classifier] in_channels={self.in_channels} but "
                    f"input has {C} channels."
                )
                logger.error(msg)
                raise ValueError(msg)

            if len(spatial) != self.spatial_dims:
                msg = (
                    f"[Swinv2Classifier] spatial_dims={self.spatial_dims} but "
                    f"input has {len(spatial)} spatial dimensions."
                )
                logger.error(msg)
                raise ValueError(msg)

            # Split input into chunks of size B and pass to encoder
            mod_vecs = []
            for m in range(n_late_fusion):
                vec_sum = None
                for p in range(
                    n_patches
                ):  # stream over patches to avoid keeping many feature maps
                    xmp = x[:, m, p].contiguous()  # (B, C, *spatial)
                    f_map = self.encoder(xmp)[
                        -1
                    ]  # (B, n_feats, *bottleneck_dims)  last-scale feature
                    v = f_map.mean(
                        dim=tuple(range(2, f_map.ndim))
                    )  # GAP -> (B, n_feats)
                    vec_sum = v if vec_sum is None else vec_sum + v
                mod_vecs.append(cast(torch.Tensor, vec_sum) / n_patches)  # (B, n_feats)

            feats = torch.stack(mod_vecs, dim=1)  # (B, n_fusion, n_feats)
            feats = self.fusion_head(feats)  # (B, in_dim)
        else:
            f_map = self.encoder(x)[-1]  # (B, n_feats, *bottleneck_dims)
            feats = f_map.mean(dim=tuple(range(2, f_map.ndim)))  # GAP -> (B, n_feats)

        return self.classification_head(feats)


@register(RK.NETWORK)
class Swinv2ClassifierMidFusion(nn.Module):
    """Swin ViT V2 classifier with mid-level fusion.

    Contrary to `Swinv2Classifier`, this model preserves the spatial dimensions
    of the feature maps, performs multimodal fusion, and then aggregates them to
    a single patch-level prediction using a bagging approach defined by the
    `aggregator` argument.

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
        # classification head
        use_proj_head: bool = True,
        proj_dim: int = 64,
        n_classes: int = 1,
        n_fusion: int = 1,
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

        # Projection head to proj_dim
        self.proj_head = (
            nn.Sequential(
                nn.Conv3d(feature_size * 16, proj_dim, 1, bias=False),
                nn.GroupNorm(
                    num_groups=8 if proj_dim >= 8 else 1, num_channels=proj_dim
                ),
                nn.GELU(),
                nn.Dropout3d(0.1),
            )
            if use_proj_head
            else nn.Identity()
        )

        # Classification head
        self.score_head = nn.Conv3d(proj_dim, n_classes, 1)

        # Mid-level fusion weights across modalities (applied on logits)
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion))

        logger.info(
            f"[Swinv2ClassifierMidFusion] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}. Heads initialized with in_dim={feature_size * 16}, "
            f"n_classes={n_classes}, n_fusion={n_fusion}."
        )

    def _encode(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and extract the last feature map (B, F, d, h, w)."""
        f_maps = encoder(x)
        if isinstance(f_maps, (tuple, list)):
            f_map = f_maps[-1]
        return f_map  # (B, n_feats, *spatial_dims)

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
        """Reshape input from (B, n_late_fusion, *spatial_dims) to (B,
        n_fusion, C, *spatial_dims)."""
        if x.ndim == self.spatial_dims + 2:  # n_late_fusion in channel_dim
            x = x.unsqueeze(2)
        elif x.ndim == self.spatial_dims + 3:  # channel_dim already in place
            pass
        else:
            msg = (
                f"[{self.__class__.__name__}] Expected input shape to be "
                f"(B, N, *spatial_dims) or (B, N, C, *spatial_dims), "
                f"but got {x.shape}."
            )
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
            msg = (
                f"[Swinv2ClassifierMidFusion] Expected input with "
                f"{3 + self.spatial_dims} dims "
                f"(B, n_fusion, C, *spatial_dims); "
                f"got {tuple(x.shape)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        B, n_fusion, C, *spatial = x.shape

        if n_fusion != self.n_fusion:
            msg = (
                f"[Swinv2ClassifierMidFusion] Expected n_fusion={self.n_fusion}, "
                f"got {n_fusion}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if C != self.in_channels:
            msg = (
                f"[Swinv2ClassifierMidFusion] in_channels={self.in_channels} but "
                f"input has {C} channels."
            )
            logger.error(msg)
            raise ValueError(msg)

        if len(spatial) != self.spatial_dims:
            msg = (
                f"[Swinv2ClassifierMidFusion] spatial_dims={self.spatial_dims} but "
                f"input has {len(spatial)} spatial dimensions."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Get cell-level scores per modality
        per_mod_logits = []
        for m in range(n_fusion):
            encoder = self.encoders[0] if self._share_encoder else self.encoders[m]
            feats = self._encode(encoder, x[:, m])  # (B, n_feats, *spatial_dims)
            proj_feats = self.proj_head(feats)  # (B, proj_dim, *spatial_dims)
            logits = self.score_head(proj_feats)  # (B, n_classes, *spatial_dims)
            per_mod_logits.append(logits)

        # Mid-level fusion on logits (weighted sum across modalities)
        w = torch.softmax(self.fusion_weights, dim=0)  # (n_fusion,)
        fused = torch.stack(
            per_mod_logits, dim=0
        )  # (n_fusion, B, n_classes, *spatial_dims)
        fused = (w[:, None, None, None, None, None] * fused).sum(
            dim=0
        )  # (B, n_classes, *spatial_dims)

        # MIL pooling to bag logits
        return self._bag_logits(fused)  # (B, n_classes)


@register(RK.NETWORK)
class Swinv2LateFusionFPNDecoder(nn.Module):
    """SwinViT v2 with lightweight FPN decoder.

    Supports multimodal input (concatenated in the channel dimension)
    via *late fusion* (sharing the encoder) or *early fusion* (sharing
    the decoder).
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
        # Fusion args
        n_late_fusion: int = 1,
        # FPN decoder args
        out_channels: int = 1,
        width: int = 32,
        norm: Literal["instance", "group", "batch"] = "instance",
        norm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        extra_swin_kwargs = extra_swin_kwargs or {}

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

        # MONAI's `SwinTransformer` returns input after patch_embed and 4-level feature maps
        in_feats = [feature_size * 2**i for i in range(len(depths) + 1)]
        self.L = len(in_feats)

        self.decoder = FPNLightDecoder3D(
            in_feats=in_feats,
            in_chans=in_channels,
            out_channels=out_channels,
            width=width,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.n_late_fusion = n_late_fusion
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.fusion_weights = nn.Parameter(
            torch.zeros(1 + self.L, n_late_fusion)
        )  # +1 for input

        logger.info(
            f"[Swinv2FPNDecoder] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, feature_size={feature_size}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}. Decoder initialized with in_feats={in_feats}, "
            f"out_channels={out_channels}, width={width}, norm={norm}, norm_kwargs={norm_kwargs}."
        )

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_mod, *spatial) -> (B, n_mod, 1, *spatial) *or* (B, n_mod, C,
        *spatial) -> unchanged."""
        if x.ndim == 2 + self.spatial_dims:  # (B, n_late_fusion, *spatial_dims)
            x = x.unsqueeze(2)  # (B, n_late_fusion, 1, *spatial_dims)
        elif x.ndim == 3 + self.spatial_dims:  # (B, n_late_fusion, C, *spatial_dims)
            pass
        else:
            msg = (
                f"[{self.__class__.__name__}] Expected input shape to be "
                f"(B, n_late_fusion, *spatial_dims) or (B, n_late_fusion, C, *spatial_dims), "
                f"but got {x.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return x

    def _encode_one(self, x1: torch.Tensor) -> list[torch.Tensor]:
        """Single encoder forward pass."""
        feats = self.encoder(x1)
        if not isinstance(feats, (list, tuple)):
            msg = f"[{self.__class__.__name__}] Encoder must return a list/tuple of multi-scale features."
            logger.error(msg)
            raise RuntimeError(msg)
        if len(feats) != self.L:
            msg = (
                f"[{self.__class__.__name__}] Expected {self.L} features from encoder, "
                f"got {len(feats)}."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return list(feats)  # [f1..fL], high->low res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, *spatial_dims).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        x = self._reshape_input(x)
        B, n_late_fusion, C, *spatial = x.shape

        if n_late_fusion != self.n_late_fusion:
            msg = (
                f"[{self.__class__.__name__}] Expected n_late_fusion={self.n_late_fusion}, "
                f"got {n_late_fusion}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if C != self.in_channels:
            msg = (
                f"[{self.__class__.__name__}] in_channels={self.in_channels} but "
                f"input has {C} channels."
            )
            logger.error(msg)
            raise ValueError(msg)

        if len(spatial) != self.spatial_dims:
            msg = (
                f"[{self.__class__.__name__}] spatial_dims={self.spatial_dims} but "
                f"input has {len(spatial)} spatial dimensions."
            )
            logger.error(msg)
            raise ValueError(msg)

        # softmax weights across modalities per scale: [levels, n_late_fusion]
        alpha = torch.softmax(self.fusion_weights, dim=1)

        # running fused features per scale; fill lazily at first modality
        fused_in: torch.Tensor | None = None
        fused_feats: list[torch.Tensor | None] = [None] * (alpha.shape[0] - 1)

        # split input into chunks of size B and pass to encoder
        for m in range(n_late_fusion):
            in_m = x[:, m]  # (B, C, *spatial)
            # add zero scale (input img)
            w = alpha[0, m]
            fused_in = fused_in + w * in_m if fused_in is not None else w * in_m
            # encode and add rest
            feats_m = self._encode_one(in_m)
            for l in range(self.L):
                w = alpha[1 + l, m]
                fused_feats[l] = (
                    fused_feats[l] + w * feats_m[l]  # type: ignore
                    if fused_feats[l] is not None  # type: ignore
                    else w * feats_m[l]
                )
            del in_m, feats_m

        return self.decoder(fused_in, fused_feats)


@register(RK.NETWORK)
class SwinMIM(SwinViT):
    """`SwinTransformer` v2 (with residual convolutions) for Masked Image
    Modeling.

    Replaces the `UnetResBlock` in each layer with a `PartialUnetResBlock`, which
    contains partial convolutions and mask-aware normalization to prevent contamination
    of feature intensities and statistics from adjacent masked patches. In addition,
    it supports a custom channel-specific patch embedding for multimodal inputs via mid-fusion.

    See `PartialUnetResBlock`, `PartialConv`, and `MaskedInstanceNorm` documentation
    for more details.

    Notes:
        - If no mask is provided, the model behaves as a regular MONAI `SwinTransformer`.
        - Masking is performed after patch embedding and before the first layer of the encoder,
          so the mask is expected to match the patch grid dimensions; i.e., (B, n_features, *spatial_dims).
        - The mask locations remain preserved throughout each level of the encoder.
        - During patch merging, the mask is downsampled using an `AND` rule; i.e., all adjacent
          patches within the merging window must be masked.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        embed_dim: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        extra_swin_kwargs: dict[str, Any] | None = None,
        merge_mode: Literal["and", "or"] = "and",
    ):
        extra_swin_kwargs = extra_swin_kwargs or {}

        super().__init__(
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
            use_v2=use_v2,
            **extra_swin_kwargs,
        )

        # Replace the `UnetResBlock` in each layer with a `PartialUnetResBlock`
        for name in ("layers1c", "layers2c", "layers3c", "layers4c"):
            layerc = getattr(self, name, None)
            if (
                layerc is None
                or not isinstance(layerc, nn.ModuleList)
                or len(layerc) == 0
            ):
                msg = f"[{self.__class__.__name__}] `{name}` missing or empty."
                logger.error(msg)
                raise ValueError(msg)

            if not isinstance(layerc[0], MONAIUnetrBasicBlock):
                msg = (
                    f"[{self.__class__.__name__}] `{name}[0]` is `{type(layerc[0])}`, "
                    f"expected `UnetrBasicBlock`."
                )
                logger.error(msg)
                raise ValueError(msg)

            layerc[0] = PartialUnetrBasicBlock(cast(MONAIUnetrBasicBlock, layerc[0]))

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.mask_tokenizer = MaskToken(embed_dim)

        if merge_mode not in ("and", "or"):
            msg = f"[{self.__class__.__name__}] merge_mode must be 'and' or 'or', got {merge_mode}."
            logger.error(msg)
            raise ValueError(msg)
        self.merge_mode = merge_mode

        logger.info(
            f"[{self.__class__.__name__}] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, embed_dim={embed_dim}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}."
        )

    @staticmethod
    def _validate_input_tensors(
        model: nn.Module,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Validate input tensor and optional mask compatibility."""
        if x.ndim != 2 + cast(int, model.spatial_dims):
            msg = (
                f"[{model.__class__.__name__}] Expected input shape to be "
                f"(B, C, *spatial_dims), but got {x.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)

        B, C, *spatial = x.shape

        if C != model.in_channels:
            msg = (
                f"[{model.__class__.__name__}] in_channels={model.in_channels} but "
                f"input has {C} channels."
            )
            logger.error(msg)
            raise ValueError(msg)

        if len(spatial) != model.spatial_dims:
            msg = (
                f"[{model.__class__.__name__}] spatial_dims={model.spatial_dims} but "
                f"input has {len(spatial)} spatial dimensions."
            )
            logger.error(msg)
            raise ValueError(msg)

        if mask is not None:
            if mask.device != x.device:
                msg = (
                    f"[{model.__class__.__name__}] mask.device does not match x.device: "
                    f"mask.device={mask.device}, x.device={x.device}."
                )
                logger.error(msg)
                raise ValueError(msg)

            if mask.ndim != 2 + cast(int, model.spatial_dims):
                msg = (
                    f"[{model.__class__.__name__}] Expected mask shape to be "
                    f"(B, embed_dim, *spatial_dims), but got {mask.shape}."
                )
                logger.error(msg)
                raise ValueError(msg)

            B_m, C_m, *_ = mask.shape

            if C_m != model.embed_dim:
                msg = (
                    f"[{model.__class__.__name__}] embed_dim={model.embed_dim} but "
                    f"mask has {C_m} channels."
                )
                logger.error(msg)
                raise ValueError(msg)

            if B_m != B:
                msg = (
                    f"[{model.__class__.__name__}] Expected same batch size for input and mask, but "
                    f"{B}, but got {B_m}."
                )
                logger.error(msg)
                raise ValueError(msg)

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
        *,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Modified mask-aware version of MONAI's `SwinTransformer` for MIM.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims).
            normalize: Whether to perform layer normalization at the output of each level.
            mask: Boolean tensor of shape (B, embed_dim, *spatial_dims) indicating masked
                features after patch embedding.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, *spatial_dims).
        """
        SwinMIM._validate_input_tensors(self, x, mask)

        if mask is None:
            return super().forward(x, normalize)

        # Level 0: Patch embedding and optional mask injection: at this level
        # mask is per-feature, to allow separate patterns in multiple modalities.
        x0 = self.patch_embed(x)
        if mask.shape != x0.shape:
            msg = (
                f"[{self.__class__.__name__}] Expected mask to match patch grid x0; "
                f"got mask={mask.shape}, x0={x0.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)
        x0 = self.mask_tokenizer.apply_mask(x0, mask)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)

        # Assign patch-wide mask label across all channels for subsequent levels
        # (B, embed_dim, *spatial_features); all 1s when all features are masked
        mask = mask.all(dim=1, keepdim=True).expand_as(x0)

        # Level 1: First residual conv + SwinTransformer block; downsampling mask
        # and replacing conv layers with partial convs.
        if self.use_v2:
            x0, m0 = self.layers1c[0](x0.contiguous(), mask=mask)
            x0 = self.mask_tokenizer.apply_mask(x0, m0)
            m1 = merge_mask_nd(
                m0, mode=self.merge_mode
            )  # mirror patch merging for subsequent level
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)

        # Level 2: Second residual conv + SwinTransformer block; same as level 1
        # for all subsequent levels.
        if self.use_v2:
            x1, m1 = self.layers2c[0](x1.contiguous(), mask=m1)
            x1 = self.mask_tokenizer.apply_mask(x1, m1)
            m2 = merge_mask_nd(m1, mode=self.merge_mode)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)

        # Level 3
        if self.use_v2:
            x2, m2 = self.layers3c[0](x2.contiguous(), mask=m2)
            x2 = self.mask_tokenizer.apply_mask(x2, m2)
            m3 = merge_mask_nd(m2, mode=self.merge_mode)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)

        # Level 4; last
        if self.use_v2:
            x3, m3 = self.layers4c[0](x3.contiguous(), mask=m3)
            x3 = self.mask_tokenizer.apply_mask(x3, m3)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]


@register(RK.NETWORK)
class SwinSimMIM(SwinViT):
    """`SwinTransformer` v2 (with residual convolutions) for SimMIM-style pre-
    training.

    Unlike `SwinMIM`, this model uses a mask token to replace masked patches
    after patch embedding but does not employ mask-aware convolutions and
    normalization.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        embed_dim: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        extra_swin_kwargs: dict[str, Any] | None = None,
    ):
        extra_swin_kwargs = extra_swin_kwargs or {}

        super().__init__(
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
            use_v2=use_v2,
            **extra_swin_kwargs,
        )

        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.mask_tokenizer = MaskToken(embed_dim)

        logger.info(
            f"[{self.__class__.__name__}] Encoder initialized with in_channels={in_channels}, "
            f"depths={depths}, num_heads={num_heads}, window_size={window_size}, "
            f"patch_size={patch_size}, embed_dim={embed_dim}, use_v2={use_v2}, "
            f"extra_swin_kwargs={extra_swin_kwargs}."
        )

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
        *,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        SwinMIM._validate_input_tensors(self, x, mask)

        if mask is None:
            return super().forward(x, normalize)

        x0 = self.patch_embed(x)
        if mask.shape != x0.shape:
            msg = (
                f"[{self.__class__.__name__}] Expected mask to match patch grid x0; "
                f"got mask={mask.shape}, x0={x0.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)
        x0 = self.mask_tokenizer.apply_mask(x0, mask)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


@register(RK.NETWORK)
class Multimodal3DSwinMIMFPN(nn.Module):
    """`SwinTransformer` v2 (with residual convolutions) for 3D Masked Image
    Modeling with a FPN decoder and multimodal patch embedding support.

    See `SwinMIM`, `FPNLightDecoder3D`, and `MultimodalPatchEmbed` for more details
    on the input arguments and `forward()` contract.

    If `use_vanilla_swin` is True, the model uses the original MONAI SwinViT encoder
    without mask-aware layers for SimMIM-style pre-training.
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
        embed_dim: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict[str, Any] | None = None,
        use_vanilla_swin: bool = False,
        merge_mode: Literal["and", "or"] = "and",
        # Multimodal patch embedding args
        inject_modality_tokens: Sequence[bool] | bool = False,
        expected_modalities: Sequence[Sequence[str]] | Sequence[str] | None = None,
        fusion: Literal["none", "conv1x1"] = "none",
        # FPN decoder args
        fpn_width: int = 32,
        fpn_norm: Literal["instance", "group", "batch"] = "instance",
        fpn_norm_kwargs: dict[str, Any] | None = None,
        # Head args
        out_channels: int | None = None,
    ):
        super().__init__()

        # Build encoder
        enc = SwinMIM if not use_vanilla_swin else SwinSimMIM
        encoder_kwargs = {
            "in_channels": in_channels,
            "patch_size": patch_size,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "embed_dim": embed_dim,
            "use_v2": use_v2,
            "spatial_dims": 3,
            "extra_swin_kwargs": extra_swin_kwargs,
        }
        if enc == SwinMIM:
            encoder_kwargs["merge_mode"] = merge_mode
        self.encoder = enc(**encoder_kwargs)

        # Multimodal patch embedding
        patch_embed = MultimodalPatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            spatial_dims=3,
            inject_modality_tokens=inject_modality_tokens,
            expected_modalities=expected_modalities,
            fusion=fusion,
        )
        self.encoder.patch_embed = MultimodalPatchEmbedAdapter(patch_embed)

        # Build decoder: FPN + voxel shuffle head
        self.fpn = FPNDecoder3DFeaturesOnly(
            in_feats=[embed_dim * 2**i for i in range(len(depths) + 1)],
            width=fpn_width,
            norm=fpn_norm,
            norm_kwargs=fpn_norm_kwargs,
        )

        self.head = VoxelShuffleHead3D(
            in_ch=fpn_width,
            out_ch=out_channels or in_channels,
            up=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
        *,
        mask: torch.Tensor | None = None,
        modality: Sequence[Sequence[str | None]] | Sequence[str | None] | None = None,
    ) -> torch.Tensor:
        cast(MultimodalPatchEmbedAdapter, self.encoder.patch_embed).set_modality(
            modality
        )
        enc_feats = self.encoder(x, normalize, mask=mask)
        fpn_feats = self.fpn(enc_feats)
        return self.head(fpn_feats)


@register(RK.NETWORK)
class LateFusion3DSwinMIMFPN(nn.Module):
    """`SwinTransformer` v2 (with residual convolutions) for 3D Masked Image
    Modeling with a multimodal patch embedding (as in `Multimodal3DSwinMIMFPN`)
    and a FPN decoder with additional late fusion support.

    Late fusion is performed by encoding each (B, C, *spatial_dims) input for each
    of the `n_late_fusion` streams and combining the encoded features at each scale
    using learnable fusion weights (`n_late_fusion` x `n_levels`).

    See `Multimodal3DSwinMIMFPN` for more details on the input arguments and
    `forward()` contract.
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
        embed_dim: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict[str, Any] | None = None,
        use_vanilla_swin: bool = False,
        merge_mode: Literal["and", "or"] = "and",
        # Multimodal patch embedding args
        inject_modality_tokens: Sequence[bool] | bool = False,
        expected_modalities: Sequence[Sequence[str]] | Sequence[str] | None = None,
        fusion: Literal["none", "conv1x1"] = "none",
        # FPN decoder args
        fpn_width: int = 32,
        fpn_norm: Literal["instance", "group", "batch"] = "instance",
        fpn_norm_kwargs: dict[str, Any] | None = None,
        # Fusion args
        n_late_fusion: int = 1,
        # Head args
        out_channels: int | None = None,
    ):
        super().__init__()

        # Build encoder
        enc = SwinMIM if not use_vanilla_swin else SwinSimMIM
        encoder_kwargs = {
            "in_channels": in_channels,
            "patch_size": patch_size,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "embed_dim": embed_dim,
            "use_v2": use_v2,
            "spatial_dims": 3,
            "extra_swin_kwargs": extra_swin_kwargs,
        }
        if enc == SwinMIM:
            encoder_kwargs["merge_mode"] = merge_mode
        self.encoder = enc(**encoder_kwargs)

        # Multimodal patch embedding
        patch_embed = MultimodalPatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            spatial_dims=3,
            inject_modality_tokens=inject_modality_tokens,
            expected_modalities=expected_modalities,
            fusion=fusion,
        )
        self.encoder.patch_embed = MultimodalPatchEmbedAdapter(patch_embed)

        # MONAI's `SwinTransformer` returns input after patch_embed and 4-level feature maps
        in_feats = [embed_dim * 2**i for i in range(len(depths) + 1)]
        self.L = len(in_feats)

        # Build decoder: FPN + voxel shuffle head
        self.fpn = FPNDecoder3DFeaturesOnly(
            in_feats=in_feats,
            width=fpn_width,
            norm=fpn_norm,
            norm_kwargs=fpn_norm_kwargs,
        )

        self.head = VoxelShuffleHead3D(
            in_ch=fpn_width,
            out_ch=out_channels or in_channels,
            up=patch_size,
        )

        self.n_late_fusion = n_late_fusion
        self.fusion_weights = nn.Parameter(torch.zeros(self.L, n_late_fusion))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_mod, *spatial) -> (B, n_mod, 1, *spatial) *or* (B, n_mod, C,
        *spatial) -> unchanged."""
        if x.ndim == 5:  # (B, n_late_fusion, *spatial_dims)
            x = x.unsqueeze(2)  # (B, n_late_fusion, 1, *spatial_dims)
        elif x.ndim == 6:  # (B, n_late_fusion, C, *spatial_dims)
            pass
        else:
            msg = (
                f"[{self.__class__.__name__}] Expected input shape to be "
                f"(B, n_late_fusion, *spatial_dims) or (B, n_late_fusion, C, *spatial_dims), "
                f"but got {x.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return x

    def _encode_one(self, x1: torch.Tensor) -> list[torch.Tensor]:
        """Single encoder forward pass."""
        feats = self.encoder(x1)

        if not isinstance(feats, (list, tuple)):
            msg = f"[{self.__class__.__name__}] Encoder must return a list/tuple of multi-scale features."
            logger.error(msg)
            raise RuntimeError(msg)

        if len(feats) != self.L:
            msg = (
                f"[{self.__class__.__name__}] Expected {self.L} features from encoder, "
                f"got {len(feats)}."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return list(feats)  # [f1..fL], high->low res

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
        *,
        mask: torch.Tensor | None = None,
        modality: Sequence[Sequence[str | None]] | Sequence[str | None] | None = None,
    ) -> torch.Tensor:
        """Input tensor is expected to be of shape (B, n_late_fusion,
        *spatial_dims) or (B, n_late_fusion, C, *spatial_dims)."""
        cast(MultimodalPatchEmbedAdapter, self.encoder.patch_embed).set_modality(
            modality
        )

        x = self._reshape_input(x)
        n_late_fusion = x.shape[1]

        if n_late_fusion != self.n_late_fusion:
            msg = (
                f"[{self.__class__.__name__}] Expected n_late_fusion={self.n_late_fusion}, "
                f"got {n_late_fusion}."
            )
            logger.error(msg)
            raise ValueError(msg)

        # softmax weights across modalities per scale: [levels, n_late_fusion]
        alpha = torch.sigmoid(self.fusion_weights)

        # running fused features per scale; fill lazily at first modality
        fused_feats: list[torch.Tensor | None] = [None] * self.L

        # split input into chunks of size B and pass to encoder
        for m in range(n_late_fusion):
            in_m = x[:, m]  # (B, C, *spatial_dims)
            feats_m = self._encode_one(in_m)
            for l in range(self.L):
                w = alpha[l, m]
                prev = fused_feats[l]
                if prev is None:
                    fused_feats[l] = w * feats_m[l]
                else:
                    fused_feats[l] = prev + w * feats_m[l]
            del in_m, feats_m

        fpn_feats = self.fpn(fused_feats)
        return self.head(fpn_feats)
