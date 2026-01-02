"""Contains masking transforms for masked autoencoder training."""

__all__ = [
    "CreateRandomMaskd",
    "SaveReconstructionTargetd",
    "CreateEmptyMaskd",
    "GetKeyQueryd",
    "SlidingWindowPatch",
    "SlidingWindowPatchd",
    "RandImgKeyd",
    "ClipNonzeroPercentilesd",
    "UnscalePredsIfNeeded",
]

from typing import Literal, Sequence, Any
from collections.abc import Hashable
import logging
import math
from copy import deepcopy

import numpy as np
import torch
from monai.data.utils import dense_patch_slices

# pyright: reportPrivateImportUsage=false
from monai.transforms.utils import TransformBackends
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
)
from monai.data import MetaTensor

from anyBrainer.core.transforms.utils import assign_key
from anyBrainer.core.utils import (
    ensure_tuple_dim,
    pad_to_size,
)

logger = logging.getLogger(__name__)


class CreateRandomMaskd(MapTransform, Randomizable):
    """Create a random 3D mask for masked autoencoder training.

    Args:
        keys: Keys to generate masks for (typically 'img')
        mask_key: Key to store the generated mask
        mask_ratio: Fraction of voxels to mask (0.0 to 1.0)
        mask_patch_size: Size of mask patches (cube side length)
        allow_missing_keys: Whether to allow missing keys
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: Sequence[str] = ("img",),
        mask_key: str = "mask",
        mask_ratio: float = 0.6,
        mask_patch_size: int = 4,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.mask_key = mask_key
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

    def randomize(self, img_shape):
        # Calculate number of patches in each dimension
        d1, d2, d3 = img_shape[-3:]
        n_patches = (
            np.array([d1, d2, d3]) + self.mask_patch_size - 1
        ) // self.mask_patch_size
        total_patches = np.prod(n_patches)

        # Sample which patches to mask
        n_masked = int(total_patches * self.mask_ratio)
        patch_mask = torch.ones(int(total_patches), dtype=torch.bool)

        idx = torch.as_tensor(
            self.R.choice(int(total_patches), n_masked, replace=False)
        )
        patch_mask[idx] = False
        self._patch_mask = patch_mask.view(*map(int, n_patches))

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            img = d[key]
            img_shape = img.shape
            try:
                d1, d2, d3 = img_shape[-3:]
            except Exception as e:
                msg = f"CreateRandomMaskd expects a 3-D volume, got {img_shape}"
                logger.error(msg)
                raise ValueError(msg) from e

            _, mt_state, pos, *_ = self.R.get_state()
            self.randomize(img_shape)

            # Vectorised up-sampling back to voxel space
            mask = (
                self._patch_mask.repeat_interleave(self.mask_patch_size, 0)
                .repeat_interleave(self.mask_patch_size, 1)
                .repeat_interleave(self.mask_patch_size, 2)
            )

            # Crop in case spatial dims are not multiples of patch size
            mask = mask[:d1, :d2, :d3]

            # Get original dims
            while mask.ndim < img.ndim:
                mask = mask.unsqueeze(0)

            logger.debug(
                f"mask shape: {mask.shape}, "
                f"mask ratio: {1 - mask.float().mean().item()}, "
                f"rng: pos={pos:3d}, first5={mt_state[:5]}"
            )

            # Keep metadata
            if isinstance(img, MetaTensor):
                d[self.mask_key] = MetaTensor(mask, meta=img.meta)
            else:
                d[self.mask_key] = mask
        return d


class SaveReconstructionTargetd(MapTransform):
    """Save a copy of the image as reconstruction target before intensity
    augmentations.

    This creates a 'recon' key with the current image data.
    """

    def __init__(
        self,
        keys: Sequence[str] | str = "img",
        recon_key: str = "recon",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.recon_key = recon_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[self.recon_key] = d[key].clone()
        return d


class CreateEmptyMaskd(MapTransform):
    """Create an empty mask."""

    def __init__(
        self,
        keys: Sequence[str] | str = "img",
        mask_key: str = "brain_mask",
        skip_if_exists: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.skip_if_exists = skip_if_exists

    def __call__(self, data):
        d = dict(data)

        if self.skip_if_exists and self.mask_key in d:
            return d

        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, MetaTensor):
                d[self.mask_key] = MetaTensor(torch.zeros_like(img), meta=img.meta)
            else:
                d[self.mask_key] = torch.zeros_like(img)
        return d


class GetKeyQueryd(MapTransform, Randomizable):
    """Create a key-query pair for contrastive learning by randomly assigning
    query and key or by always assigning augmented view to key.

    Random assignment does not prevent key-query pair from being the same.
    Can specify additional extra keys that can be either query/key-specific or fixed.

    Args:
        keys_prefix: Prefix for the img keys to be used for the key-query pair.
        count_key: Key for the number of modalities.
        extra_iters: Prefix for the extra keys to be used for the key-query pair.
        always_augment_query: Whether to always augment the query.
        query_key: Key for the query.
        extra_keys: Non-iterable extra keys to be used for the key-query pair.
        track: Whether to track the query and key indices.
    """

    def __init__(
        self,
        keys_prefix: str | None = "img",
        count_key: str | None = "count",
        extra_iters: Sequence[str] | None = ["mod"],
        always_augment_query: bool = False,
        query_key: str | None = None,
        extra_keys: Sequence[str] | None = ["sub_id", "ses_id"],
        track: bool = False,
    ) -> None:

        if always_augment_query and query_key is None:
            msg = "query_key must be provided if always_augment_query is True"
            logger.error(msg)
            raise ValueError(msg)

        if always_augment_query and keys_prefix is not None:
            logger.warning("keys_prefix is ignored if always_augment_query is True")

        if keys_prefix is None and always_augment_query == False:
            msg = "keys_prefix must be provided if always_augment_query is False"
            logger.error(msg)
            raise ValueError(msg)

        if keys_prefix is not None and count_key is None:
            msg = "count_key must be provided if keys_prefix is provided"
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(keys="img", allow_missing_keys=False)
        self.keys_prefix = keys_prefix
        self.count_key = count_key
        self.extra_iters = extra_iters
        self.always_augment_query = always_augment_query
        self.query_key = query_key
        self.extra_keys = extra_keys
        self.track = track

    def randomize(self, count: int):
        query_idx, key_idx = self.R.randint(0, count, size=2)
        return query_idx, key_idx

    def __call__(self, data):
        d = dict(data)
        new_d = {}

        # Typical contrastive; fixed key for query, augmented view for key
        if self.always_augment_query:
            new_d["query"] = assign_key(d, self.query_key)
            new_d["key"] = assign_key(d, self.query_key)

            if self.extra_keys is not None:
                for key in self.extra_keys:
                    new_d[key] = assign_key(d, key)
            return new_d

        # Randomly assign query and key
        if self.count_key not in d:
            msg = f"count_key {self.count_key} not found in data"
            logger.error(msg)
            raise ValueError(msg)

        query_idx, key_idx = self.randomize(d[self.count_key])
        new_d["query"] = assign_key(d, f"{self.keys_prefix}_{query_idx}")
        new_d["key"] = assign_key(d, f"{self.keys_prefix}_{key_idx}")

        # Assign query/key-specific extra keys; e.g. modality
        if self.extra_iters is not None:
            for key in self.extra_iters:
                new_d[key] = assign_key(d, f"{key}_{query_idx}")

        if self.extra_keys is not None:
            for key in self.extra_keys:
                new_d[key] = assign_key(d, key)

        if self.track:
            new_d["track"] = {
                "query_idx": query_idx,
                "key_idx": key_idx,
            }
        return new_d


class SlidingWindowPatch(Transform):
    """Extracts sliding window patches from a tensor of shape (C,
    *spatial_dims), returning a tensor of shape (N_patches, C, *patch_size).

    Stride of the sliding window is calculated as either by `overlap`
    or automatically to match `n_patches`.

    If `n_patches` is provided, `overlap` is ignored.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        patch_size: int | Sequence[int],
        overlap: float | Sequence[float] | None = 0.5,
        n_patches: int | Sequence[int] | None = None,
        padding_mode: str = "constant",
        padding_side: Literal["right", "left", "both"] = "both",
        spatial_dims: int = 3,
    ):
        """
        Args:
            patch_size: Target patch size.
            overlap: Overlap between patches.
            n_patches: Number of patches to extract.
            padding_mode: Padding mode; see `torch.nn.functional.pad`.
            padding_side: Side to pad; can be 'right', 'left', or 'both'.
            spatial_dims: Number of spatial dimensions.
        """
        super().__init__()
        if n_patches is None and overlap is None:
            msg = "Either `n_patches` or `overlap` must be provided"
            logger.error(msg)
            raise ValueError(msg)
        if n_patches is not None and overlap is not None:
            logger.warning(
                "Provided both `n_patches` and `overlap`; will use `n_patches`."
            )

        self.patch_size = ensure_tuple_dim(patch_size, dim=spatial_dims)
        self.overlap = (
            ensure_tuple_dim(overlap, dim=spatial_dims) if overlap is not None else None
        )
        self.n_patches = (
            ensure_tuple_dim(n_patches, dim=spatial_dims)
            if n_patches is not None
            else None
        )

        self.padding_mode = padding_mode
        self.padding_side = padding_side
        self.spatial_dims = spatial_dims
        self.slices: list[tuple[slice, ...]] = []  # will be populated by __call__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() < 1 + self.spatial_dims:
            msg = (
                f"Expected at least {self.spatial_dims + 1}D tensor "
                f"(C, *spatial), got shape {img.shape}"
            )
            logger.error(msg)
            raise ValueError(msg)

        C, *spatial = img.shape[-self.spatial_dims - 1 :]
        min_target_dims = [
            max(S, P) for S, P in zip(spatial, self.patch_size)
        ]  # minimum target dim to pad

        # Calculate stride
        if self.n_patches is not None:
            stride = []
            target_dims = []
            for S, P, N in zip(min_target_dims, self.patch_size, self.n_patches):
                if N <= 1:
                    st = max(1, S)
                    target_dim = S
                else:
                    st = max(1, math.ceil(max(0, S - P) / (N - 1)))
                    target_dim = (
                        N - 1
                    ) * st + P  # possibly extend padding to force N patches
                stride.append(st)
                target_dims.append(target_dim)
        else:
            stride = tuple(
                max(1, int(round(p * (1.0 - o))))
                for p, o in zip(self.patch_size, self.overlap)
            )  # type: ignore[arg-type]
            target_dims = min_target_dims

        # Pad to target dims
        img = pad_to_size(
            img,
            target=target_dims,
            spatial_dims=self.spatial_dims,
            mode=self.padding_mode,
            side=self.padding_side,
        )  # type: ignore[arg-type]
        spatial = img.shape[-self.spatial_dims :]  # updated spatial dims after padding

        # Get slices for patches
        self.slices = dense_patch_slices(spatial, self.patch_size, stride)

        prefix = (slice(None),) * (
            img.ndim - self.spatial_dims - 1
        )  # leading dims before C
        patches = [img[prefix + (slice(None),) + slc] for slc in self.slices]

        return torch.stack(
            patches, dim=len(prefix)
        )  # shape: (..., N_patches, C, *patch_size)


class SlidingWindowPatchd(MapTransform):
    """Dictionary-based version of `SlidingWindowPatch`.

    See `SlidingWindowPatch` for more details.
    """

    def __init__(
        self,
        keys: Sequence[str] | str = "img",
        patch_size: int | Sequence[int] = 128,
        overlap: float | Sequence[float] | None = 0.5,
        n_patches: int | Sequence[int] | None = None,
        padding_mode: str = "constant",
        padding_side: Literal["right", "left", "both"] = "right",
        spatial_dims: int = 3,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.transform = SlidingWindowPatch(
            patch_size=patch_size,
            overlap=overlap,
            n_patches=n_patches,
            padding_mode=padding_mode,
            padding_side=padding_side,
            spatial_dims=spatial_dims,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class RandImgKeyd(MapTransform, Randomizable):
    """Dictionary transform to randomly select a key from the provided keys and
    store the value in a new key."""

    def __init__(
        self,
        keys: Sequence[Hashable] | Hashable,
        new_key: str = "img",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.new_key = new_key

    def randomize(self, candidates: Sequence[Hashable]) -> Hashable:
        idx = int(self.R.randint(0, len(candidates)))
        return candidates[idx]

    def __call__(self, data):
        d = dict(data)
        candidates = [k for k in self.key_iterator(d)]
        if not candidates:
            if self.allow_missing_keys:
                return d
            msg = f"None of {self.keys} present in data."
            logger.error(msg)
            raise KeyError(msg)

        chosen_key = self.randomize(candidates)
        d[self.new_key] = deepcopy(d[chosen_key])
        return d


class ClipNonzeroPercentilesd(MapTransform):
    """Clip values per-channel to [lower, upper] percentiles computed over
    nonzero voxels.

    Works with torch.Tensor and monai.data.MetaTensor on any device.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, keys, lower=0.5, upper=99.9, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        assert 0 <= lower < upper <= 100
        self.q = (lower / 100.0, upper / 100.0)

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = d[k]  # (C, ...), torch.Tensor or MetaTensor
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            # per-channel
            for c in range(x.shape[0]):
                xc = x[c]
                nz = xc != 0
                if nz.any():
                    # work in float for quantiles; keep device/meta
                    vals = xc[nz].to(dtype=torch.float32)
                    lo, hi = torch.quantile(
                        vals, torch.tensor(self.q, device=vals.device)
                    )
                    # in-place clamp only on nonzeros
                    xc[nz] = torch.clamp(xc[nz], min=lo.item(), max=hi.item())
            d[k] = x  # same object; MetaTensor meta preserved
        return d


class UnscalePredsIfNeeded(Transform):
    """Array transform to reverse scaling and centering applied to regression
    predictions.

    Usage (array):     t = UnscalePredsIfNeeded(meta={"scale_range":
    (min_val, max_val), "center": center})     unscaled = t(pred)
    """

    def __init__(
        self,
        scale_range: tuple[Any, Any] | None = None,
        center: int | float | None = None,
    ):
        """
        Args:
            scale_range: (min_val, max_val) for scaling pred to [-1, 1]
            center: offset to add to pred to center them around 0
        """
        self.scale_range = scale_range
        self.center = center

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` must be a torch.Tensor, got {type(pred)}")

        out = pred

        # Unscale
        if self.scale_range is not None:
            min_val, max_val = self.scale_range
            # keep device/dtype consistent and allow per-element tensors
            min_t = torch.as_tensor(min_val, dtype=out.dtype, device=out.device)
            max_t = torch.as_tensor(max_val, dtype=out.dtype, device=out.device)
            scale = (max_t - min_t) / 2
            out = out * scale

        # Uncenter
        if self.center is not None:
            center_t = torch.as_tensor(self.center, dtype=out.dtype, device=out.device)
            out = out + center_t

        return out
