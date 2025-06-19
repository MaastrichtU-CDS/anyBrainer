"""
Contains masking transforms for masked autoencoder training.
"""

__all__ = [
    "CreateRandomMaskd",
    "SaveReconstructionTargetd",
]

from typing import Sequence
import logging

import numpy as np
import torch

# pyright: reportPrivateImportUsage=false
from monai.transforms.utils import TransformBackends
from monai.transforms import (
    MapTransform,
    Randomizable,
)
from monai.data import MetaTensor

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
            
            # Calculate number of patches in each dimension
            n_patches = (np.array([d1, d2, d3]) + self.mask_patch_size - 1) // self.mask_patch_size
            total_patches = np.prod(n_patches)

            # Sample which patches to mask
            n_masked = int(total_patches * self.mask_ratio)
            patch_mask = torch.ones(int(total_patches), dtype=torch.bool)

            last_rng = self.R.get_state()[1][:5] # pyright: ignore[reportArgumentType]
            idx = torch.as_tensor(
                self.R.choice(int(total_patches), n_masked, replace=False)
            )
            patch_mask[idx] = False
            patch_mask = patch_mask.view(*map(int, n_patches))

            # Vectorised up-sampling back to voxel space
            mask = patch_mask.repeat_interleave(self.mask_patch_size, 0) \
                                 .repeat_interleave(self.mask_patch_size, 1) \
                                 .repeat_interleave(self.mask_patch_size, 2)                  

            # Crop in case spatial dims are not multiples of patch size
            mask = mask[:d1, :d2, :d3]

            # Get original dims
            while mask.ndim < img.ndim:
                mask = mask.unsqueeze(0)
                
            logger.debug(f"mask shape: {mask.shape}, "
                         f"mask ratio: {1 - mask.float().mean().item()}, "
                         f"rng: {last_rng}")
 
            # Keep metadata
            if isinstance(img, MetaTensor):
                d[self.mask_key] = MetaTensor(mask, meta=img.meta, affine=img.affine)
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
        keys: Sequence[str] = ("img",), 
        recon_key: str = "recon",
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.recon_key = recon_key
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            # Create a copy for reconstruction target
            d[self.recon_key] = d[key].copy()
        return d

