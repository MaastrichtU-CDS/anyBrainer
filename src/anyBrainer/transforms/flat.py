"""Composed MONAI transforms"""

__all__ = [
    'get_mae_train_transforms',
    'get_mae_val_transforms',
    'get_contrastive_train_transforms',
    'get_contrastive_val_transforms',
]

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
)

from .masking_transforms import (
    SaveReconstructionTargetd, 
    CreateRandomMaskd,
)

def get_mae_train_transforms():
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), 
                   mode='constant', allow_missing_keys=True),
        RandFlipd(keys=['img', 'brain_mask'], spatial_axis=(0, 1), prob=0.3, 
                  allow_missing_keys=True),
        RandAffined(keys=['img', 'brain_mask'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode=['bilinear', 'nearest'], padding_mode='zeros', prob=1.0, 
                    allow_missing_keys=True),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128), 
                         allow_missing_keys=True),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
        RandScaleIntensityFixedMeand(keys=['img'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['img'], std=0.01, prob=0.2),
        RandGaussianSmoothd(keys=['img'], sigma_x=(0.5, 1.0), prob=0.2),
        RandBiasFieldd(keys=['img'], coeff_range=(0.0, 0.05), prob=0.3),
        RandGibbsNoised(keys=['img'], alpha=(0.2, 0.4), prob=0.2),
        RandAdjustContrastd(keys=['img'], gamma=(0.9, 1.1), prob=0.3),
        RandSimulateLowResolutiond(keys=['img'], prob=0.1, zoom_range=(0.5, 1.0)),
    ]

def get_mae_val_transforms():
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), 
                   mode='constant', allow_missing_keys=True),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128), 
                         allow_missing_keys=True),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
    ]

def get_contrastive_train_transforms():
    return [
        LoadImaged(keys=['query', 'key'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['query', 'key'], spatial_size=(128, 128, 128), mode='constant'),
        RandFlipd(keys=['query'], spatial_axis=(0, 1), prob=0.3),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['query'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=(128, 128, 128)),
        RandSpatialCropd(keys=['key'], roi_size=(128, 128, 128)),
        RandScaleIntensityFixedMeand(keys=['query'], factors=0.1, prob=0.3),
        RandScaleIntensityFixedMeand(keys=['key'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['query'], std=0.01, prob=0.2),
        RandGaussianNoised(keys=['key'], std=0.01, prob=0.2),
        RandGaussianSmoothd(keys=['query'], sigma_x=(0.5, 1.0), prob=0.2),
        RandGaussianSmoothd(keys=['key'], sigma_x=(0.5, 1.0), prob=0.2),
        RandBiasFieldd(keys=['query'], coeff_range=(0.0, 0.05), prob=0.3),
        RandBiasFieldd(keys=['key'], coeff_range=(0.0, 0.05), prob=0.3),
        RandGibbsNoised(keys=['query'], alpha=(0.2, 0.4), prob=0.2),
        RandGibbsNoised(keys=['key'], alpha=(0.2, 0.4), prob=0.2),
        RandAdjustContrastd(keys=['query'], gamma=(0.9, 1.1), prob=0.3),
        RandAdjustContrastd(keys=['key'], gamma=(0.9, 1.1), prob=0.3),
        RandSimulateLowResolutiond(keys=['query'], prob=0.1, zoom_range=(0.5, 1.0)),
        RandSimulateLowResolutiond(keys=['key'], prob=0.1, zoom_range=(0.5, 1.0)),
    ]

def get_contrastive_val_transforms():
    return [
        LoadImaged(keys=['query', 'key'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['query', 'key'], spatial_size=(128, 128, 128), mode='constant'),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=(128, 128, 128)),
        RandSpatialCropd(keys=['key'], roi_size=(128, 128, 128)),
        RandScaleIntensityFixedMeand(keys=['key'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['key'], std=0.01, prob=0.2),
        RandGaussianSmoothd(keys=['key'], sigma_x=(0.5, 1.0), prob=0.2),
        RandBiasFieldd(keys=['key'], coeff_range=(0.0, 0.05), prob=0.3),
        RandGibbsNoised(keys=['key'], alpha=(0.2, 0.4), prob=0.2),
        RandAdjustContrastd(keys=['key'], gamma=(0.9, 1.1), prob=0.3),
        RandSimulateLowResolutiond(keys=['key'], prob=0.1, zoom_range=(0.5, 1.0)),
    ]