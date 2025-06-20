"""
Transform managers that orchestrate builders based on mode.
"""

__all__ = [
    "ContrastiveTransformManager",
    "MAETransformManager",
]

from typing import Dict, Any, Sequence
import logging

# pyright: reportPrivateImportUsage=false
from monai.transforms import Compose, Transform

from .interfaces import TransformManagerInterface
from .builders import (
    LoadTransformBuilder,
    SpatialTransformBuilder, 
    IntensityTransformBuilder,
    MaskingTransformBuilder
)

logger = logging.getLogger(__name__)


class ContrastiveTransformManager(TransformManagerInterface):
    """Transform manager for contrastive learning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        load_config = self.config.get('load_transforms', False)
        if not load_config:
            logger.error("Missing load_transforms settings.")
            raise ValueError("Missing load_transforms settings.")
        
        spatial_config_1 = self.config.get('spatial_transforms', {})
        intensity_config_1 = self.config.get('intensity_transforms', {})

        # Check if different augmentations for second view
        spatial_config_2 = self.config.get('spatial_transforms_2', spatial_config_1)
        intensity_config_2 = self.config.get('intensity_transforms_2', intensity_config_1)
        
        # Get patch size
        patch_size_1 = self.config.get('patch_size', (128, 128, 128))
        patch_size_2 = self.config.get('patch_size_2', patch_size_1)

        # Initialize transform builders for first view
        self.load_composer_1 = LoadTransformBuilder(load_config.update({'patch_size': patch_size_1}))
        self.spatial_composer_1 = SpatialTransformBuilder(spatial_config_1.update({'patch_size': patch_size_1}))
        self.intensity_composer_1 = IntensityTransformBuilder(intensity_config_1)
        
        # Initialize transforms builders for second view
        self.load_composer_2 = LoadTransformBuilder(load_config.update({'patch_size': patch_size_2}))
        self.spatial_composer_2 = SpatialTransformBuilder(spatial_config_2.update({'patch_size': patch_size_2}))
        self.intensity_composer_2 = IntensityTransformBuilder(intensity_config_2)

        # Determine img_keys from config
        max_modalities = self.config.get('max_modalities', 100)
        self.img_keys = [f"img_{i}" for i in range(max_modalities)]

    def get_train_transforms(self) -> tuple[Transform, Transform]:
        """Get training transforms for contrastive learning."""
        allow_missing_keys = True

        # Get transforms for first view
        view_1_transforms = [
            self.load_composer_1.build(self.img_keys, allow_missing_keys),
            self.spatial_composer_1.build(self.img_keys, allow_missing_keys),
            self.intensity_composer_1.build(self.img_keys, allow_missing_keys),
        ]

        # Get transforms for second view
        view_2_transforms = [
            self.load_composer_2.build(self.img_keys, allow_missing_keys),
            self.spatial_composer_2.build(self.img_keys, allow_missing_keys),
            self.intensity_composer_2.build(self.img_keys, allow_missing_keys),
        ]
        
        return Compose(view_1_transforms), Compose(view_2_transforms)

    def get_val_transforms(self) -> tuple[Transform, Transform]:
        """Get validation transforms for contrastive learning."""
        allow_missing_keys = True

        # Get transforms for first view
        view_1_transforms = [self.load_composer_1.build(self.img_keys, allow_missing_keys)]

        # Get transforms for second view
        view_2_transforms = [self.load_composer_2.build(self.img_keys, allow_missing_keys)]
        
        return Compose(view_1_transforms), Compose(view_2_transforms)


class MAETransformManager(TransformManagerInterface):
    """Transform manager for masked autoencoder."""

    # MAE-specific configuration
    BRAIN_MASK_INTERPOLATION = 'nearest'
    BRAIN_MASK_PADDING = 'zeros'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        load_config = self.config.get('load_transforms', False)
        if not load_config:
            logger.error("Missing load_transforms settings.")
            raise ValueError("Missing load_transforms settings.")
        
        spatial_config = self.config.get('spatial_transforms', {})
        masking_config = self.config.get('mae_transforms', {})
        intensity_config = self.config.get('intensity_transforms', {})

        # Get patch size
        patch_size = self.config.get('patch_size', (128, 128, 128))
        load_config.update({'patch_size': patch_size})
        spatial_config.update({'patch_size': patch_size})

        # Set any custom configs
        spatial_config = self._configure_mae_spatial_transforms(spatial_config)

        # Initialize transform builders
        self.load_composer = LoadTransformBuilder(load_config)
        self.spatial_composer = SpatialTransformBuilder(spatial_config)
        self.masking_composer = MaskingTransformBuilder(masking_config)
        self.intensity_composer = IntensityTransformBuilder(intensity_config)
    
    def get_train_transforms(self) -> Transform:
        """
        Get training transforms for masked autoencoder. 

        The mask autoencoder transforms will create two extra keys; 
        one for the reconstruction target ('recon') and one for masking ('mask').
        """
        return Compose([
            self.load_composer.build(['img', 'brain_mask'], allow_missing_keys=True),
            self.spatial_composer.build(['img', 'brain_mask'], allow_missing_keys=True),
            self.masking_composer.build(['img'], allow_missing_keys=True),
            self.intensity_composer.build(['img'], allow_missing_keys=False), # only apply to img key
        ])
            
    def get_val_transforms(self) -> Transform:
        """Get validation transforms for masked autoencoder."""
        return Compose([
            self.load_composer.build(['img', 'brain_mask'], allow_missing_keys=True),
            self.masking_composer.build(['img'], allow_missing_keys=True),
        ])
        
    def _configure_mae_spatial_transforms(self, spatial_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom spatial config logic for MAE.

        The user interpolation and padding settings will be applied to the image key.
        Internally, the affine transform will be applied to the image and brain_mask keys, 
        with nearest interpolation and zeros padding for the brain_mask key.
        """

        if 'affine' in spatial_config:
            # Allow user configuration only for the image key.
            usr_interp_setting = spatial_config['affine'].get('mode', 'bilinear')
            usr_padding_setting = spatial_config['affine'].get('padding_mode', 'border')
            
            # Update the affine config
            spatial_config['affine'].update({'mode': [usr_interp_setting, self.BRAIN_MASK_INTERPOLATION], 
                                             'padding_mode': [usr_padding_setting, self.BRAIN_MASK_PADDING]})
        return spatial_config