"""
Contains masking transforms for masked autoencoder training.
"""

__all__ = [
    "CreateRandomMaskd",
    "SaveReconstructionTargetd",
    "CreateEmptyMaskd",
    "GetKeyQueryd",
    "SlidingWindowPatch",
    "SlidingWindowPatchd",
]

from typing import Sequence
import logging

import torch.nn.functional as F
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

from anyBrainer.core.transforms.utils import (
    assign_key
)
from anyBrainer.core.utils.misc import (
    ensure_tuple_dim,
)

logger = logging.getLogger(__name__)


class CreateRandomMaskd(MapTransform, Randomizable):
    """
    Create a random 3D mask for masked autoencoder training.
    
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
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.mask_key = mask_key
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
    
    def randomize(self, img_shape):
        # Calculate number of patches in each dimension
        d1, d2, d3 = img_shape[-3:]
        n_patches = (np.array([d1, d2, d3]) + self.mask_patch_size - 1) // self.mask_patch_size
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
            mask = self._patch_mask.repeat_interleave(self.mask_patch_size, 0) \
                                 .repeat_interleave(self.mask_patch_size, 1) \
                                 .repeat_interleave(self.mask_patch_size, 2)                  

            # Crop in case spatial dims are not multiples of patch size
            mask = mask[:d1, :d2, :d3]

            # Get original dims
            while mask.ndim < img.ndim:
                mask = mask.unsqueeze(0)
            
            logger.debug(f"mask shape: {mask.shape}, "
                         f"mask ratio: {1 - mask.float().mean().item()}, "
                         f"rng: pos={pos:3d}, first5={mt_state[:5]}")
 
            # Keep metadata
            if isinstance(img, MetaTensor):
                d[self.mask_key] = MetaTensor(mask, meta=img.meta)
            else:
                d[self.mask_key] = mask
        return d


class SaveReconstructionTargetd(MapTransform):
    """
    Save a copy of the image as reconstruction target before intensity augmentations.
    This creates a 'recon' key with the current image data.
    """
    def __init__(
        self, 
        keys: Sequence[str] | str = "img", 
        recon_key: str = "recon",
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.recon_key = recon_key
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[self.recon_key] = d[key].clone()
        return d


class CreateEmptyMaskd(MapTransform):
    """
    Create an empty mask.
    """
    def __init__(
        self, 
        keys: Sequence[str] | str = "img", 
        mask_key: str = "brain_mask", 
        skip_if_exists: bool = True,
        allow_missing_keys: bool = False
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
                d[self.mask_key] = MetaTensor(
                    torch.zeros_like(img), meta=img.meta
                )
            else:
                d[self.mask_key] = torch.zeros_like(img)
        return d


class GetKeyQueryd(MapTransform, Randomizable):
    """
    Create a key-query pair for contrastive learning by randomly assigning
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
        new_d['query'] = assign_key(d, f"{self.keys_prefix}_{query_idx}")
        new_d['key'] = assign_key(d, f"{self.keys_prefix}_{key_idx}")
        
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
    """
    Extracts sliding window patches from a tensor of shape (C, *spatial_dims),
    returning a tensor of shape (N_patches, C, *patch_size).
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        patch_size: int | Sequence[int],
        overlap: float | Sequence[float] = 0.5,
        padding_mode: str = "constant",
        spatial_dims: int = 3,
    ):
        super().__init__()
        self.patch_size = ensure_tuple_dim(patch_size, dim=spatial_dims)
        self.overlap = ensure_tuple_dim(overlap, dim=spatial_dims)
        self.padding_mode = padding_mode
        self.spatial_dims = spatial_dims

        self.slices: list[tuple[slice, ...]] = [] # will be populated by __call__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() < 1 + self.spatial_dims:
            msg = (f"Expected at least {self.spatial_dims + 1}D tensor "
                   f"(C, *spatial), got shape {img.shape}")
            logger.error(msg)
            raise ValueError(msg)
        
        C, *spatial = img.shape[-self.spatial_dims - 1:]

        # Convert relative overlap to stride
        scan_interval = tuple(max(1, int(round(p * (1.0 - o))))
                              for p, o in zip(self.patch_size, self.overlap))

        # Pad if necessary
        pad_needed = [
            (max(p - s, 0), 0) for p, s in zip(reversed(self.patch_size), reversed(spatial))
        ]
        if any(p[0] > 0 for p in pad_needed):
            pad_flat = [i for pair in pad_needed for i in pair]
            img = F.pad(img, pad=pad_flat, mode=self.padding_mode)
            spatial = img.shape[-self.spatial_dims:]

        # Get slices for patches
        self.slices = dense_patch_slices(spatial, self.patch_size, scan_interval)

        prefix = (slice(None),) * (img.ndim - self.spatial_dims - 1)  # leading dims before C
        patches = [img[prefix + (slice(None),) + slc] for slc in self.slices]

        return torch.stack(patches, dim=len(prefix))  # shape: (..., N_patches, C, *patch_size)


class SlidingWindowPatchd(MapTransform):
    """
    Dictionary-based version of `SlidingWindowPatch`.

    See `SlidingWindowPatch` for more details.
    """
    def __init__(
        self,
        keys: Sequence[str] | str = "img",
        patch_size: int | Sequence[int] = 128,
        overlap: float | Sequence[float] = 0.5,
        padding_mode: str = "constant",
        spatial_dims: int = 3,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.transform = SlidingWindowPatch(
            patch_size=patch_size,
            overlap=overlap,
            padding_mode=padding_mode,
            spatial_dims=spatial_dims,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
            logger.info(f"[{self.__class__.__name__}] {key} shape: {d[key].shape}")
        return d