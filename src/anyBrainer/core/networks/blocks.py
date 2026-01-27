"""Pytorch blocks of layers to be used in the construction of networks."""

__all__ = [
    "ProjectionHead",
    "ClassificationHead",
    "FusionHead",
    "ConvNormAct3d",
    "FPNLightDecoder3D",
    "FPNDecoder3DFeaturesOnly",
    "VoxelShuffleHead3D",
    "MaskToken",
    "PartialConv",
    "MaskedInstanceNorm",
    "MaskedInstanceNorm3d",
    "PartialUnetResBlock",
    "PartialUnetrBasicBlock",
    "MultimodalPatchEmbed",
    "MultimodalPatchEmbedAdapter",
]

import logging
from typing import Any, Literal, Sequence, overload, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution as MONAIConv
from monai.networks.blocks.dynunet_block import UnetResBlock as MONAIUnetResBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock as MONAIUnetrBasicBlock
from monai.networks.blocks.patchembedding import PatchEmbed

from anyBrainer.core.networks.utils import (
    get_mlp_head_args,
    voxel_shuffle_3d,
    upsample_to_3d,
    center_pad_mask_like,
)
from anyBrainer.core.utils import ensure_tuple_dim
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
    ) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if activation_kwargs is None:
            activation_kwargs = {}

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            UnitFactory.get_activation_from_kwargs(
                {"name": activation, **activation_kwargs}
            ),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )

        # Log hyperparameters
        msg = (
            f"[{self.__class__.__name__}] ProjectionHead initialized with in_dim={in_dim}, "
            f"hidden_dim={hidden_dim}, proj_dim={proj_dim}"
        )
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
    ) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        if activation_kwargs is None:
            activation_kwargs = {}

        dropouts, hidden_dims, activations, activation_kwargs = get_mlp_head_args(
            in_dim,
            num_classes,
            num_hidden_layers,
            dropout,
            hidden_dim,
            activation,
            activation_kwargs,
        )

        layers = []
        for i in range(num_hidden_layers + 1):
            layers.extend(
                [
                    nn.Dropout(p=dropouts[i]),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    UnitFactory.get_activation_from_kwargs(
                        {"name": activations[i], **activation_kwargs[i]}
                    ),
                ]
            )

        self.classifier = nn.Sequential(*layers)

        # Log hyperparameters
        summary = (
            f"in_dim={in_dim}, num_classes={num_classes}, "
            f"num_hidden_layers={num_hidden_layers}, dropout(s)={dropouts}, "
            f"linear_layer(s)={hidden_dims}, activation(s)={activations}"
        )
        logger.info(
            f"[{self.__class__.__name__}] ClassificationHead initialized with {summary}"
        )

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
    """Unify representation across brain patches and different modalities.

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
    ) -> None:
        super().__init__()
        self.n_fusion = n_fusion
        self.mod_only = mod_only
        self.fusion_weights = nn.Parameter(torch.ones(n_fusion))

        logger.info(
            f"[{self.__class__.__name__}] FusionHead initialized with "
            f"n_fusion={n_fusion}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.mod_only:  # Fuse everything
            if x.dim() == 7:  # (B, n_fusion, n_patches, n_features, D, H, W)
                pass
            elif x.dim() == 6:  # (B, n_fusion (or n_patches), n_features, D, H, W)
                x = x.unsqueeze(1)
            elif x.dim() == 5:  # (B, n_features, D, H, W)
                x = x.unsqueeze(1).unsqueeze(2)
            else:
                msg = f"[{self.__class__.__name__}] Unexpected input shape {x.shape}"
                logger.error(msg)
                raise ValueError(msg)
            x = x.mean(dim=[-3, -2, -1])  # (B, n_fusion, n_patches, n_features)
            x = x.mean(dim=2)  # (B, n_fusion, n_features)
        else:  # Already fused, except modalities
            if x.dim() != 3:
                msg = (
                    f"[{self.__class__.__name__}] Invalid input shape {x.shape} "
                    f"for `mod_only` mode; expected (B, n_fusion, n_features)."
                )
                logger.error(msg)
                raise ValueError(msg)

        if x.size(1) != self.n_fusion:
            msg = (
                f"[{self.__class__.__name__}] Unexpected number of fusion channels; "
                f"expected {self.n_fusion}, got {x.size(1)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        weights = torch.softmax(self.fusion_weights, dim=0)  # (n_fusion,)
        x = (x * weights.view(1, -1, 1)).sum(dim=1)  # (B, n_features)
        return x


class ConvNormAct3d(nn.Module):
    """
    Convolutional block with normalization and activation applied
    in the order: conv -> norm -> act.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        norm: Literal["instance", "group", "batch"] = "instance",
        act: Literal["relu", "gelu", "leaky_relu"] = "gelu",
        norm_kwargs: dict[str, Any] | None = None,
        act_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            k: Kernel size
            s: Stride
            p: Padding
            norm: Normalization layer
            act: Activation function
            norm_kwargs: Keyword arguments for the normalization layer
            act_kwargs: Keyword arguments for the activation function
        """
        super().__init__()
        # Conv
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
        )

        # Norm
        norm_kwargs = norm_kwargs or {}
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_ch, **norm_kwargs)
        elif norm == "group":
            n = norm_kwargs.pop("num_groups", 8)
            self.norm = nn.GroupNorm(
                num_groups=min(n, out_ch), num_channels=out_ch, **norm_kwargs
            )
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_ch, **norm_kwargs)
        else:
            raise ValueError(f"Invalid normalization layer: {norm}")

        # Act
        act_kwargs = act_kwargs or {}
        if act == "relu":
            self.act = nn.ReLU(**act_kwargs)
        elif act == "gelu":
            self.act = nn.GELU(**act_kwargs)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(**act_kwargs)
        else:
            raise ValueError(f"Invalid activation function: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FPNLightDecoder3D(nn.Module):
    """Top-down FPN with narrow width and one 3x3 smooth per level.

    in_chans: channels of encoder features [c1,c2,c3,c4,c5] (f1 highest res)
    width: lateral width (e.g., 32 or 64)
    """

    def __init__(
        self,
        in_feats: Sequence[int],
        in_chans: int,
        out_channels: int,
        width: int = 32,
        norm: Literal["instance", "group", "batch"] = "instance",
        norm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        assert len(in_feats) == 5, "Need 5 scales [f1..f5]"
        f1, f2, f3, f4, f5 = in_feats

        self.in_stem = ConvNormAct3d(
            in_chans, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )

        self.lat1 = nn.Conv3d(f1, width, kernel_size=1, bias=False)
        self.lat2 = nn.Conv3d(f2, width, kernel_size=1, bias=False)
        self.lat3 = nn.Conv3d(f3, width, kernel_size=1, bias=False)
        self.lat4 = nn.Conv3d(f4, width, kernel_size=1, bias=False)
        self.lat5 = nn.Conv3d(f5, width, kernel_size=1, bias=False)

        # one light smooth per pyramid level after lateral+topdown add
        self.smooth4 = ConvNormAct3d(
            width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )
        self.smooth3 = ConvNormAct3d(
            width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )
        self.smooth2 = ConvNormAct3d(
            width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )
        self.smooth1 = ConvNormAct3d(
            width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )
        self.smooth0 = ConvNormAct3d(
            width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
        )

        # head on the finest map (keeps params tiny)
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-3:], mode="trilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [f1,f2,f3,f4,f5] high->low res
        if len(feats) != 5:
            msg = (
                f"[{self.__class__.__name__}] Expected 5 features, "
                f"got {len(feats)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        f1, f2, f3, f4, f5 = feats
        p5 = self.lat5(f5)  # bottleneck
        p4 = self.smooth4(self.lat4(f4) + self._upsample_to(p5, f4))
        p3 = self.smooth3(self.lat3(f3) + self._upsample_to(p4, f3))
        p2 = self.smooth2(self.lat2(f2) + self._upsample_to(p3, f2))
        p1 = self.smooth1(self.lat1(f1) + self._upsample_to(p2, f1))
        p0 = self.smooth0(self.in_stem(x) + self._upsample_to(p1, x))
        return self.head(p0)  # logits at full res


class FPNDecoder3DFeaturesOnly(nn.Module):
    """FPN that fuses 5 encoder scales (high->low: f1..f5) into a narrow
    feature map at f1 resolution."""

    def __init__(
        self,
        in_feats: Sequence[int],
        width: int = 32,
        norm: Literal["instance", "group", "batch"] = "instance",
        norm_kwargs: dict[str, Any] | None = None,
        use_smooth: bool = True,
    ) -> None:
        super().__init__()
        assert len(in_feats) == 5, "Need 5 scales [f1..f5]"
        f1, f2, f3, f4, f5 = in_feats

        # Laterals (1x1x1) to common width
        self.lat1 = nn.Conv3d(f1, width, kernel_size=1, bias=False)
        self.lat2 = nn.Conv3d(f2, width, kernel_size=1, bias=False)
        self.lat3 = nn.Conv3d(f3, width, kernel_size=1, bias=False)
        self.lat4 = nn.Conv3d(f4, width, kernel_size=1, bias=False)
        self.lat5 = nn.Conv3d(f5, width, kernel_size=1, bias=False)

        # Optional light 3x3x3 smoothing after top-down adds
        if use_smooth:
            self.smooth4 = ConvNormAct3d(
                width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
            )
            self.smooth3 = ConvNormAct3d(
                width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
            )
            self.smooth2 = ConvNormAct3d(
                width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
            )
            self.smooth1 = ConvNormAct3d(
                width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs
            )
        else:
            self.smooth4 = self.smooth3 = self.smooth2 = self.smooth1 = nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: list of 5 tensors [f1,f2,f3,f4,f5] with shapes
                   f1: (B,C1,D1,H1,W1) ... f5: (B,C5,D5,H5,W5), where
                   D1,H1,W1 are the finest token grid (typically image/P).
        Returns:
            p1: fused features at f1 resolution, shape (B, width, D1, H1, W1)
        """
        if len(feats) != 5:
            raise ValueError(
                f"[{self.__class__.__name__}] Expected 5 features, got {len(feats)}."
            )
        f1, f2, f3, f4, f5 = feats

        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + upsample_to_3d(p5, f4)
        p4 = self.smooth4(p4)

        p3 = self.lat3(f3) + upsample_to_3d(p4, f3)
        p3 = self.smooth3(p3)

        p2 = self.lat2(f2) + upsample_to_3d(p3, f2)
        p2 = self.smooth2(p2)

        p1 = self.lat1(f1) + upsample_to_3d(p2, f1)
        p1 = self.smooth1(p1)

        return p1


class VoxelShuffleHead3D(nn.Module):
    """Voxel shuffle head for 3D tensors.

    Applies a 1x1x1 convolution to the get target number of channels and
    then gets original spatial dimensions back via voxel shuffle.
    """

    def __init__(self, in_ch: int, out_ch: int, up: Sequence[int] | int) -> None:
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            up: Upscale factor; set equal to ViT patch size
        """
        super().__init__()
        self.up = ensure_tuple_dim(up, 3)
        target_dims = out_ch * self.up[0] * self.up[1] * self.up[2]
        self.proj = nn.Conv3d(in_ch, target_dims, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return voxel_shuffle_3d(self.proj(x), self.up)


class MaskToken(nn.Module):
    """Apply mask token to input tensor for masked image modeling.

    Also adjusts mask token to match input tensor channels if needed to
    account for expanding channel dimension throughout the network.
    """

    def __init__(self, embed_dim: int, std: float = 0.02) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        with torch.no_grad():
            nn.init.trunc_normal_(self.mask_token, std=std)

        logger.info(
            f"[{self.__class__.__name__}] MaskToken initialized with embed_dim={embed_dim}"
        )

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, *spatial)  channels-first
        mask: (B, C, *spatial)  bool
        """
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        if mask.shape != x.shape:
            msg = f"[{self.__class__.__name__}] mask must match x; got {mask.shape} vs {x.shape}"
            logger.error(msg)
            raise ValueError(msg)

        C = x.shape[1]
        base = self.mask_token.numel()

        tok_1d = self.mask_token
        if base != C:
            if C % base != 0:
                msg = f"mask_token has {base} channels but x has {C}; not divisible."
                logger.error(msg)
                raise ValueError(msg)
            tok_1d = tok_1d.repeat_interleave(C // base)

        tok = tok_1d.view(1, C, *([1] * (x.dim() - 2)))
        return torch.where(mask, tok, x)


class PartialConv(nn.Module):
    """Partial convolution for MIM-style pre-training to avoid mixing masked
    and unmasked patch features.

    Wraps MONAI's `monai.networks.blocks.convolution.Convolution` and
    renormalizes outputs before `ADN` the block (if present) to ensure
    no shrinkage of valid activations near mask boundaries.

    Note:
        Mask is expected to be per-channel; spatial validity is `any` over channels,
        invalid only if all channels are masked.
    """

    def __init__(self, conv: MONAIConv) -> None:
        super().__init__()

        if not hasattr(conv, "conv"):
            msg = (
                f"[{self.__class__.__name__}] Expected MONAI-style `conv` "
                f"(e.g., `monai.networks.blocks.convolution.Convolution`) to "
                f"expose internal conv as `self.conv`."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Get internal convolution attributes from the MONAI-style conv
        self.conv = getattr(conv, "conv")
        self.spatial_dims = self.conv.weight.dim() - 2
        self.kernel_size = ensure_tuple_dim(self.conv.kernel_size, self.spatial_dims)
        self.stride = ensure_tuple_dim(self.conv.stride, self.spatial_dims)
        self.padding = ensure_tuple_dim(self.conv.padding, self.spatial_dims)
        self.dilation = ensure_tuple_dim(self.conv.dilation, self.spatial_dims)

        # Get ADN (if present)
        self.adn = getattr(conv, "adn", None)

        # Total number of elements in the kernel for renormalization
        K = 1
        for ki in self.kernel_size:
            K *= int(ki)
        self.register_buffer(
            "_K", torch.tensor(float(K), dtype=torch.float32), persistent=False
        )

        # Ones kernel for counting valid elements
        self.register_buffer(
            "ones_kernel",
            torch.ones(1, 1, *self.kernel_size, dtype=torch.float32),
            persistent=False,
        )

    def _get_valid_counts(self, mask: torch.Tensor) -> torch.Tensor:
        """Get number of contributing elements (i.e., counts) for
        renormalization."""
        # validity: 1 if at least one channel is unmasked at that spatial site
        # shape: (B,1,*spatial)
        valid = (~mask).any(dim=1, keepdim=True).to(dtype=torch.float32)

        if self.spatial_dims == 3:
            return F.conv3d(
                valid,
                self.ones_kernel,  # type: ignore
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        if self.spatial_dims == 2:
            return F.conv2d(
                valid,
                self.ones_kernel,  # type: ignore
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
        if self.spatial_dims == 1:
            return F.conv1d(
                valid,
                self.ones_kernel,  # type: ignore
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        msg = (
            f"[{self.__class__.__name__}] Unsupported spatial_dims={self.spatial_dims}"
        )
        logger.error(msg)
        raise ValueError(msg)

    def forward(
        self, x: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            y = self.conv(x)
            if self.adn is not None:
                y = self.adn(y)
            return y

        if mask.device != x.device:
            msg = (
                f"[{self.__class__.__name__}] mask.device does not match x.device: "
                f"mask.device={mask.device}, x.device={x.device}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if mask.shape != x.shape:
            msg = (
                f"[{self.__class__.__name__}] mask.shape does not match x.shape: "
                f"mask={mask.shape}, x={x.shape}."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Replace masked values with 0
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        x = x.masked_fill(mask, 0)

        # Conv only (no ADN yet)
        y = self.conv(x)

        # Get scale for renormalization
        counts = self._get_valid_counts(mask).clamp_min(1e-6)
        scale = (self._K / counts).to(dtype=y.dtype)  # type: ignore

        # Get bias for renormalization
        bias = self.conv.bias
        if bias is not None:
            # Reshape bias to broadcast: (1, C_out, 1, 1, 1...) depending on spatial dims
            shape = [1, -1] + [1] * self.spatial_dims
            b = bias.view(*shape).to(dtype=y.dtype)
            y = (y - b) * scale + b
        else:
            y = y * scale

        # ADN (if present)
        if self.adn is not None:
            y = self.adn(y)

        return y


class MaskedInstanceNorm(nn.Module):
    """Mask-aware InstanceNorm for MIM-style pre-training.

    Computes per-sample, per-channel mean/var over valid spatial sites only.

    Note:
        Mask is expected to be per-channel; spatial validity is `any` over channels,
        invalid only if all channels are masked.
    """

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        *,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            num_features: Number of features (channels) in the input tensor.
            eps: Epsilon to avoid division by zero.
            affine: Whether to learn scale and bias parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _get_valid_mean_var(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and variance over valid spatial sites only."""
        # validity: 1 if at least one channel is unmasked at that spatial site
        # shape: (B,1,*spatial)
        spatial_dims = tuple(range(2, x.ndim))
        valid = (~mask).any(dim=1, keepdim=True).to(dtype=x.dtype)
        n_valid = valid.sum(dim=spatial_dims, keepdim=True).clamp_min(1.0)
        mean = (x * valid).sum(dim=spatial_dims, keepdim=True) / n_valid
        xc = (x - mean) * valid
        var = (xc * xc).sum(dim=spatial_dims, keepdim=True) / n_valid
        return mean, var

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim < 3:
            msg = f"[{self.__class__.__name__}] Expected (B, C, *spatial_dims), got {x.shape}."
            logger.error(msg)
            raise ValueError(msg)

        B, C, *spatial_dims = x.shape
        if mask is None:
            mean = x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)
            var = x.var(dim=tuple(range(2, x.ndim)), keepdim=True, unbiased=False)
            y = (x - mean) / torch.sqrt(var + self.eps)
        else:
            if mask.device != x.device:
                msg = (
                    f"[{self.__class__.__name__}] mask.device does not match x.device: "
                    f"mask.device={mask.device}, x.device={x.device}."
                )
                logger.error(msg)
                raise ValueError(msg)

            if mask.shape != x.shape:
                msg = (
                    f"[{self.__class__.__name__}] Expected mask to match input spatial dimensions, "
                    f"but got mask={mask.shape} and x={x.shape}."
                )
                logger.error(msg)
                raise ValueError(msg)

            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            mean, var = self._get_valid_mean_var(x, mask)

            y = (x - mean) / torch.sqrt(var + self.eps)
            y = y.masked_fill(mask, 0.0)

        if self.affine:
            if C != self.num_features:
                msg = (
                    f"[{self.__class__.__name__}] Expected {self.num_features} channels, "
                    f"but got {C}."
                )
                logger.error(msg)
                raise ValueError(msg)

            shape = [1, -1] + [1] * len(spatial_dims)
            y = y * self.weight.view(*shape) + self.bias.view(*shape)
            if mask is not None:
                y = y.masked_fill(mask, 0.0)

        return y


class MaskedInstanceNorm3d(MaskedInstanceNorm):
    pass


class PartialUnetResBlock(nn.Module):
    """Partial U-Net residual block for MIM-style pre-training.

    Wraps MONAI's `monai.networks.blocks.dynunet_block.UnetResBlock` and
    replaces the convolutions with partial convolutions to avoid mixing
    masked and unmasked patch features.
    """

    def __init__(self, unet_res_block: MONAIUnetResBlock) -> None:
        super().__init__()

        if not isinstance(unet_res_block, MONAIUnetResBlock):
            msg = (
                f"[{self.__class__.__name__}] Expected MONAI `unet_res_block` to be an instance "
                f"of `UnetResBlock`, got {type(unet_res_block)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Swap convolutions with partial convolutions
        self.conv1 = PartialConv(unet_res_block.conv1)
        self.conv2 = PartialConv(unet_res_block.conv2)
        if hasattr(unet_res_block, "conv3"):
            self.conv3 = PartialConv(unet_res_block.conv3)

        # Sanity checks for partial convolutions
        convs = [self.conv1, self.conv2]
        if hasattr(unet_res_block, "conv3"):
            convs.append(self.conv3)

        for conv in convs:
            # stride must be all 1s; otherwise simple shape alignment for mask will fail
            if any(s != 1 for s in conv.stride):
                msg = f"[{self.__class__.__name__}] Expected stride (1, 1, 1), got {conv.conv.stride}."
                logger.error(msg)
                raise ValueError(msg)

            # avoid double application of act/dropout/norm
            if conv.adn is not None:
                msg = (
                    f"[{self.__class__.__name__}] Expected wrapped MONAI conv to have no ADN "
                    f"(no act/norm/dropout), but found adn={type(conv.adn)}."
                )
                logger.error(msg)
                raise ValueError(msg)

        # Swap norm layers with mask-aware instance norm
        self.norm1 = MaskedInstanceNorm(num_features=self.conv1.conv.out_channels)
        self.norm2 = MaskedInstanceNorm(num_features=self.conv2.conv.out_channels)
        if hasattr(unet_res_block, "norm3"):
            self.norm3 = MaskedInstanceNorm(num_features=self.conv3.conv.out_channels)

        self.lrelu = unet_res_block.lrelu

    @overload
    def forward(self, inp: torch.Tensor, *, mask: None) -> torch.Tensor: ...

    @overload
    def forward(
        self, inp: torch.Tensor, *, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        inp: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = inp
        mask_residual = mask

        # conv1 + norm1 + lrelu
        out = self.conv1(inp, mask=mask)
        if mask is not None:
            mask = center_pad_mask_like(mask, out)
        out = self.norm1(out, mask=mask)
        out = self.lrelu(out)

        # conv2 + norm2
        out = self.conv2(out, mask=mask)
        if mask is not None:
            mask = center_pad_mask_like(mask, out)
        out = self.norm2(out, mask=mask)

        # residual: optional conv3 + norm3
        if hasattr(self, "conv3"):
            residual = self.conv3(residual, mask=mask_residual)
            if mask_residual is not None:
                mask_residual = center_pad_mask_like(mask_residual, residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual, mask=mask_residual)

        # Defensive check
        if mask is not None and mask_residual is not None:
            if not mask.equal(mask_residual):
                msg = (
                    f"[{self.__class__.__name__}] Expected mask and mask_residual to be the same, "
                    f"but got mask={mask.shape} and mask_residual={mask_residual.shape}."
                )
                logger.error(msg)
                raise RuntimeError(msg)

        out += residual
        out = self.lrelu(out)

        return (out, mask) if mask is not None else out


class PartialUnetrBasicBlock(nn.Module):
    """Partial UNETR residual block for MIM-style pre-training.

    Wraps MONAI's `UnetrBasicBlock` for drop-in replacement.
    Internally replaces its `.layer` (a MONAI `UnetResBlock`) with `PartialUnetResBlock`.

    Forward contract:
      - If mask is None: returns Tensor (drop-in compatible)
      - If mask is provided: returns (Tensor, mask_out)
    """

    def __init__(self, unetr_basic_block: MONAIUnetrBasicBlock) -> None:
        super().__init__()
        if not isinstance(unetr_basic_block, MONAIUnetrBasicBlock):
            msg = (
                f"[{self.__class__.__name__}] Expected MONAI `unetr_basic_block` to be an instance "
                f"of `UnetrBasicBlock`, got {type(unetr_basic_block)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(unetr_basic_block.layer, MONAIUnetResBlock):
            msg = (
                f"[{self.__class__.__name__}] Expected MONAI `unetr_basic_block.layer` to be an instance "
                f"of `UnetResBlock`, got {type(unetr_basic_block.layer)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.layer = PartialUnetResBlock(unet_res_block=unetr_basic_block.layer)

    @overload
    def forward(self, inp: torch.Tensor, *, mask: None = None) -> torch.Tensor: ...

    @overload
    def forward(
        self, inp: torch.Tensor, *, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        inp: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.layer(inp, mask=mask)


class MultimodalPatchEmbed(nn.Module):
    """Multimodal patch embedding for SwinMIM.

    Wraps MONAI's `PatchEmbed` to encode seperately each channel and optionally
    introduce a modality-specific token for each channel-modality combination,
    in case more than a single modality is expected per channel.

    Notes:
        - Output is channel-first: (B, embed_dim, *patch_grid).
    """

    def __init__(
        self,
        patch_size: int | Sequence[int],
        in_chans: int,
        embed_dim: int,
        spatial_dims: int = 3,
        *,
        inject_modality_tokens: Sequence[bool] | bool = False,
        expected_modalities: Sequence[Sequence[str]] | Sequence[str] | None = None,
        fusion: Literal["none", "conv1x1"] = "none",
    ) -> None:
        """
        Args:
            patch_size: Patch size for the patch embedding.
            in_chans: Number of input channels.
            embed_dim: Embedding dimension.
            spatial_dims: Spatial dimensions of the input tensor.
            inject_modality_tokens: Whether to inject modality tokens for each channel.
            expected_modalities: Expected modalities for each channel.
            fusion: Fusion mode for the modality tokens. Supports no fusion, or 1x1 convolution.
        """
        super().__init__()

        if embed_dim % in_chans != 0:
            msg = (
                f"[{self.__class__.__name__}] Expected embed_dim to be divisible by in_chans, "
                f"but got embed_dim={embed_dim} and in_chans={in_chans}."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.in_channels = in_chans
        self.embed_dim = embed_dim
        self.embed_dim_per_ch = self.embed_dim // in_chans
        self.patch_size = ensure_tuple_dim(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims

        inject = ensure_tuple_dim(inject_modality_tokens, in_chans)
        self.inject_modality_tokens = tuple(bool(v) for v in inject)

        if any(self.inject_modality_tokens):
            if expected_modalities is None:
                msg = (
                    f"[{self.__class__.__name__}] `expected_modalities` must be provided when "
                    f"inject_modality_tokens has any True."
                )
                logger.error(msg)
                raise ValueError(msg)

            if (
                not isinstance(expected_modalities, (list, tuple))
                or len(expected_modalities) != in_chans
            ):
                msg = f"[{self.__class__.__name__}] `expected_modalities` must be a list/tuple of length in_chans."
                logger.error(msg)
                raise ValueError(msg)

            # Normalize `expected_modalities` to list-of-sets per channel
            if all(isinstance(m, str) for m in expected_modalities):
                self.expected_modalities = [{m} for m in expected_modalities]
            elif all(isinstance(m, (list, tuple)) for m in expected_modalities):
                self.expected_modalities = [set(m) for m in expected_modalities]
            else:
                msg = (
                    f"[{self.__class__.__name__}] `expected_modalities` must be a sequence of strings "
                    f"or sequences of strings."
                )
                logger.error(msg)
                raise ValueError(msg)

        # One PatchEmbed per channel
        self.patch_embeds = nn.ModuleList(
            [
                PatchEmbed(
                    patch_size=self.patch_size,
                    in_chans=1,
                    embed_dim=self.embed_dim_per_ch,
                    spatial_dims=spatial_dims,
                )
                for _ in range(in_chans)
            ]
        )

        # Build per-channel embedding layers and mod->idx mappings for efficient lookup
        self._mod_to_idx: dict[int, dict[str, int]] = {}
        self.mod_embeddings = nn.ModuleDict()

        if expected_modalities is not None:
            for i in range(in_chans):
                if not self.inject_modality_tokens[i]:
                    continue

                mods = sorted(expected_modalities[i])
                self._mod_to_idx[i] = {mod: idx for idx, mod in enumerate(mods)}

                emb = nn.Embedding(len(mods), self.embed_dim_per_ch)
                nn.init.trunc_normal_(emb.weight, std=0.02)
                self.mod_embeddings[f"ch{i}"] = emb

        # Optional mid-fusion projection
        if fusion == "conv1x1":
            conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
            self.fusion = conv(embed_dim, embed_dim, kernel_size=1, bias=True)
        elif fusion == "none":
            self.fusion = nn.Identity()
        else:
            msg = f"[{self.__class__.__name__}] Unknown fusion: {fusion}."
            logger.error(msg)
            raise ValueError(msg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        modality: Sequence[str | None] | Sequence[Sequence[str | None]] | None = None,
    ) -> torch.Tensor:
        """Iterate over input channels and apply a modality-aware patch
        embedding.

        The `modality` entry is interpreted in three ways, depending on the context:
            - Sequence[Sequence[str | None]]: Assuming multiple channels, with the length of
                the sequence expected to match the input channels. Inner sequences are
                expected to match the batch size.
            - Sequence[str | None]: Assuming single channel; the length of the sequence is
                expected to match the batch size.
            - None: No modality is provided.

        Notes:
            - If a modality token is injected for any channel, then the `modality` input must match
              the number of all input channels, not just the number of channels with modality tokens.
              for indexing purposes. In such channels, users can set the modality to `None`.
            - No modality checking is performed for channels with no injection of modality tokens.
              Otherwise, the validity of per-channel modality entries is checked against the
              `expected_modalities` setting.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims).
            modality: Sequence of modality names for each channel.

        Returns:
            torch.Tensor: Output tensor of shape (B, embed_dim, *patch_grid).
        """
        if x.ndim != 2 + self.spatial_dims:
            msg = f"[{self.__class__.__name__}] Expected (B,C,*spatial), got {x.shape}."
            logger.error(msg)
            raise ValueError(msg)

        B, C, *_ = x.shape
        if C != self.in_channels:
            msg = (
                f"[{self.__class__.__name__}] Expected C={self.in_channels}, got C={C}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if any(self.inject_modality_tokens):
            if not isinstance(modality, (list, tuple)) or len(modality) == 0:
                msg = (
                    f"[{self.__class__.__name__}] a non-empty `modality` sequence must be provided when "
                    f"`inject_modality_tokens` is True for at least one channel; got {modality}."
                )
                logger.error(msg)
                raise ValueError(msg)

            # Convert any Sequence[str | None] to Sequence[Sequence[str | None]]
            if all(isinstance(m, (str, type(None))) for m in modality):
                modality = [cast(Sequence[str | None], modality)]

            # Check validity of Sequence[Sequence[str | None]] entries
            if (
                len(modality) != C
                or not all(isinstance(m, (list, tuple)) for m in modality)
                or any(
                    len(m) != B for m in cast(Sequence[Sequence[str | None]], modality)
                )
            ):
                msg = (
                    f"[{self.__class__.__name__}] Expected `modality` to match the number of input channels {C}, "
                    f"and comprise only sequences of length {B} for each channel."
                )
                logger.error(msg)
                raise ValueError(msg)

        outs: list[torch.Tensor] = []

        for i in range(C):
            yi = self.patch_embeds[i](x[:, i : i + 1])

            # Add modality token for the channel only if configured in construction
            if self.inject_modality_tokens[i]:
                ch_mods = cast(Sequence[Sequence[str | None]], modality)[i]
                mod_to_idx = self._mod_to_idx[i]

                # Validate and build modality tensor for `ch{i}` as (B, 1), where `1` corresponds to
                # the modality index
                indices = []
                for j, mod_j in enumerate(ch_mods):
                    if mod_j is None:
                        msg = f"[{self.__class__.__name__}] Expected modality for channel {i}, batch {j}, but got None."
                        logger.error(msg)
                        raise ValueError(msg)

                    if mod_j not in mod_to_idx:
                        msg = (
                            f"[{self.__class__.__name__}] Unknown modality '{mod_j}' for channel {i}. "
                            f"Available: {list(mod_to_idx.keys())}"
                        )
                        logger.error(msg)
                        raise ValueError(msg)

                    indices.append(mod_to_idx[mod_j])

                mod_tensor = torch.tensor(indices, device=yi.device, dtype=torch.long)

                # Lookup `ch{i}` modality tokens for each batch item
                all_toks = self.mod_embeddings[f"ch{i}"](
                    mod_tensor
                )  # (B, embed_dim_per_ch)

                # Reshape for broadcasting: (B, embed_dim_per_ch, 1, 1, ...)
                view = (B, -1) + (1,) * self.spatial_dims
                yi = yi + all_toks.view(*view).to(dtype=yi.dtype)

            outs.append(yi)

        y = torch.cat(outs, dim=1)  # (B, embed_dim, *grid)
        return self.fusion(y)


class MultimodalPatchEmbedAdapter(PatchEmbed):
    """Adapter that enables drop-in replacement of the `MultimodalPatchEmbed`
    module in the `SwinTransformer` base class.

    This is done by removing the need for a `modality` argument in the `forward()` method
    of the `MultimodalPatchEmbed` module and subclassing `PatchEmbed`.
    """

    def __init__(self, mm_embed: MultimodalPatchEmbed) -> None:
        if not isinstance(mm_embed, MultimodalPatchEmbed):
            msg = (
                f"[{self.__class__.__name__}] Expected `mm_embed` to be an "
                f"instance of `MultimodalPatchEmbed`, got {type(mm_embed)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(
            patch_size=mm_embed.patch_size,
            in_chans=mm_embed.in_channels,
            embed_dim=mm_embed.embed_dim,
            spatial_dims=mm_embed.spatial_dims,
        )
        self.mm_embed = mm_embed
        self._modality: Sequence[Sequence[str | None]] | Sequence[str | None] | None = (
            None
        )

    def set_modality(
        self,
        modality: Sequence[Sequence[str | None]] | Sequence[str | None] | None = None,
    ) -> None:
        self._modality = modality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mm_embed(x, modality=self._modality)