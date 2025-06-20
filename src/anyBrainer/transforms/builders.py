"""
Concrete transform builders implementing the interface.
"""

__all__ = [
    "LoadTransformBuilder",
    "SpatialTransformBuilder",
    "IntensityTransformBuilder",
    "MaskingTransformBuilder",
]

from typing import Sequence
import logging

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    LoadImaged,
    SpatialPadd,
    RandFlipd, 
    RandAffined, 
    RandSpatialCropd, 
    RandSimulateLowResolutiond,
    RandScaleIntensityFixedMeand, 
    RandGaussianNoised, 
    RandGaussianSmoothd,
    RandBiasFieldd, 
    RandGibbsNoised, 
    RandAdjustContrastd,
    Transform,
)

from .masking_transforms import (
    SaveReconstructionTargetd, 
    CreateRandomMaskd,
)
from .interfaces import TransformBuilderInterface

logger = logging.getLogger(__name__)


class LoadTransformBuilder(TransformBuilderInterface):
    """Builder for loading and basic preprocessing transforms."""
    
    def _build_transforms(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        patch_size = self.config.get('patch_size', (128, 128, 128))

        transforms = []
        
        transform_map = {
            'load': (LoadImaged, {
                'reader': 'NumpyReader',
                'ensure_channel_first': True,
            }),
            'pad': (SpatialPadd, {
                'spatial_size': patch_size,
                'mode': 'constant',
            }),
        }
        
        for transform_name, (transform_class, default_params) in transform_map.items():
            if self.config.get(transform_name, {}).get('enabled', False):
                params = {
                    'keys': img_keys,
                    'allow_missing_keys': allow_missing_keys,
                }
                params.update(default_params)
                params.update(self.config.get(transform_name, {}).get('params', {}))
                transforms.append(transform_class(**params))

                self.update_params(transform_name, params) # good for tracking transform settings
        
        return Compose(transforms)


class SpatialTransformBuilder(TransformBuilderInterface):
    """Builder for spatial augmentation transforms."""
    
    def _build_transforms(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        patch_size = self.config.get('patch_size', (128, 128, 128))
        transforms = []
        
        transform_map = {
            'flip': (RandFlipd, {'prob': 0.3, 'spatial_axis': [0, 1]}),
            'affine': (RandAffined, {
                'prob': 0.3,
                'rotate_range': (0.1, 0.1, 0.1),
                'scale_range': (0.1, 0.1, 0.1),
                'shear_range': (0.05, 0.05, 0.05),
                'mode': 'bilinear',
                'padding_mode': 'zeros',
            }),
            'crop': (RandSpatialCropd, {'roi_size': patch_size}),
            'low_res': (RandSimulateLowResolutiond, {'prob': 0.1, 'zoom_range': (0.5, 1.0)}),
        }
        
        for transform_name, (transform_class, default_params) in transform_map.items():
            if self.config.get(transform_name, {}).get('enabled', False):
                params = {
                    'keys': img_keys,
                    'allow_missing_keys': allow_missing_keys,
                }
                params.update(default_params)
                params.update(self.config.get(transform_name, {}).get('params', {}))
                transforms.append(transform_class(**params))

                self.update_params(transform_name, params)
        
        return Compose(transforms)


class IntensityTransformBuilder(TransformBuilderInterface):
    """Builder for intensity augmentation transforms."""
    
    def _build_transforms(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        transforms = []
        
        transform_map = {
            'scale_intensity': (RandScaleIntensityFixedMeand, {'factors': 0.1, 'prob': 0.3}),
            'gaussian_noise': (RandGaussianNoised, {'std': 0.01, 'prob': 0.2}),
            'gaussian_smooth': (RandGaussianSmoothd, {'sigma_x': (0.5, 1.0), 'prob': 0.2}),
            'bias_field': (RandBiasFieldd, {'coeff_range': (0.0, 0.05), 'prob': 0.3}),
            'gibbs_noise': (RandGibbsNoised, {'alpha': (0.2, 0.4), 'prob': 0.2}),
            'adjust_contrast': (RandAdjustContrastd, {'gamma': (0.9, 1.1), 'prob': 0.3}),
        }
        
        for transform_name, (transform_class, default_params) in transform_map.items():
            if self.config.get(transform_name, {}).get('enabled', False):
                params = {
                    'keys': img_keys,
                    'allow_missing_keys': allow_missing_keys,
                }
                params.update(default_params)
                params.update(self.config.get(transform_name, {}).get('params', {}))
                transforms.append(transform_class(**params))

                self.update_params(transform_name, params)
        
        return Compose(transforms)


class MaskingTransformBuilder(TransformBuilderInterface):
    """Builder for MAE-specific masking transforms."""
    
    def _build_transforms(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        transforms = []
        
        transform_map = {
            'reconstruction_target': (SaveReconstructionTargetd, {
                'recon_key': 'recon',
            }),
            'random_mask': (CreateRandomMaskd, {
                'mask_key': 'mask',
                'mask_ratio': 0.6,
                'mask_patch_size': 4,
            }),
        }
        
        for transform_name, (transform_class, default_params) in transform_map.items():
            if self.config.get(transform_name, {}).get('enabled', False):
                params = {
                    'keys': img_keys,
                    'allow_missing_keys': allow_missing_keys,
                }
                params.update(default_params)
                params.update(self.config.get(transform_name, {}).get('params', {}))
                transforms.append(transform_class(**params))

                self.update_params(transform_name, params)
        
        return Compose(transforms)