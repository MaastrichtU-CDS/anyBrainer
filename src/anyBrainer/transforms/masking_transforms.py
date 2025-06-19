"""
Contains masking transforms for masked autoencoder training.
"""

__all__ = [
    "CreateRandomMaskd",
    "CreateIgnoreMaskd",
    "SaveReconstructionTargetd",
]

from typing import Sequence
import logging

import numpy as np

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    MapTransform,
    Randomizable,
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
    def __init__(
        self,
        keys: Sequence[str] = ("img",),
        mask_key: str = "mask", 
        mask_ratio: float = 0.6,
        mask_patch_size: int = 4,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        Randomizable.__init__(self)
        self.mask_key = mask_key
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        
    def randomize(self, data=None) -> None:
        """Randomize method required by Randomizable interface."""
        pass
        
    def __call__(self, data):
        d = dict(data)
        self.randomize(data)
        
        for key in self.key_iterator(d):
            img_shape = d[key].shape  # Assuming C, H, W, D
            if len(img_shape) == 4:
                _, h, w, depth = img_shape
            elif len(img_shape) == 3:
                h, w, depth = img_shape
            else:
                raise ValueError(f"Expected 3D or 4D image, got shape {img_shape}")
            
            # Calculate number of patches in each dimension
            n_patches_h = h // self.mask_patch_size
            n_patches_w = w // self.mask_patch_size  
            n_patches_d = depth // self.mask_patch_size
            total_patches = n_patches_h * n_patches_w * n_patches_d
            
            # Calculate number of patches to mask
            n_masked_patches = int(total_patches * self.mask_ratio)
            
            # Create mask at patch level
            patch_mask = np.zeros(total_patches, dtype=bool)
            masked_indices = self.R.choice(total_patches, n_masked_patches, replace=False)
            patch_mask[masked_indices] = True
            
            # Reshape to 3D patch grid
            patch_mask_3d = patch_mask.reshape(n_patches_h, n_patches_w, n_patches_d)
            
            # Expand to full resolution
            mask = np.zeros((h, w, depth), dtype=bool)
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    for k in range(n_patches_d):
                        if patch_mask_3d[i, j, k]:
                            h_start = i * self.mask_patch_size
                            h_end = min((i + 1) * self.mask_patch_size, h)
                            w_start = j * self.mask_patch_size
                            w_end = min((j + 1) * self.mask_patch_size, w)
                            d_start = k * self.mask_patch_size
                            d_end = min((k + 1) * self.mask_patch_size, depth)
                            mask[h_start:h_end, w_start:w_end, d_start:d_end] = True
            
            # Add channel dimension if needed to match image shape
            if len(img_shape) == 4:
                mask = np.expand_dims(mask, axis=0)
            
            d[self.mask_key] = mask.astype(np.float32)
            
        return d


class CreateIgnoreMaskd(MapTransform):
    """
    Create an 'ignore' mask to identify background voxels (including padded regions)
    that should be excluded from loss computation.
    
    Args:
        keys: Keys to analyze for background detection
        ignore_key: Key to store the ignore mask
        background_threshold: Threshold below which voxels are considered background
        allow_missing_keys: Whether to allow missing keys
    """
    def __init__(
        self,
        keys: Sequence[str] = ("img",),
        ignore_key: str = "ignore",
        background_threshold: float = 1e-6,
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.ignore_key = ignore_key
        self.background_threshold = background_threshold
        
    def __call__(self, data):
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # Identify background voxels (close to zero or exactly zero)
            ignore_mask = np.abs(img) <= self.background_threshold
            
            # Convert to float32 for consistency
            d[self.ignore_key] = ignore_mask.astype(np.float32)
            
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

