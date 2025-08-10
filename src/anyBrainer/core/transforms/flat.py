"""Composed MONAI transforms"""

__all__ = [
    'get_mae_train_transforms',
    'get_mae_val_transforms',
    'get_contrastive_train_transforms',
    'get_contrastive_val_transforms',
    'get_predict_transforms',
    'get_classification_train_transforms',
    'get_classification_val_transforms',
]

from typing import Sequence, Callable

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    DeleteItemsd,
    ConcatItemsd,
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
    RandRotated,
)

from .unit_transforms import (
    SaveReconstructionTargetd, 
    CreateRandomMaskd,
    CreateEmptyMaskd,
    GetKeyQueryd,
    SlidingWindowPatchd,
)

from anyBrainer.registry import register, RegistryKind as RK

OPEN_KEYS = [f"img_{i}" for i in range(0, 20)]

@register(RK.TRANSFORM)
def get_mae_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
) -> list[Callable]:
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        CreateEmptyMaskd(mask_key='brain_mask'),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=patch_size, 
                   mode='constant'),
        RandFlipd(keys=['img', 'brain_mask'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['img', 'brain_mask'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode=['bilinear', 'nearest'], padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=patch_size),
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

@register(RK.TRANSFORM)
def get_mae_val_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
) -> list[Callable]:
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        CreateEmptyMaskd(mask_key='brain_mask'),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=patch_size, 
                   mode='constant'),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=patch_size),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
    ]

@register(RK.TRANSFORM)
def get_contrastive_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
) -> list[Callable]:
    return [
        GetKeyQueryd(keys_prefix='img', count_key='count', extra_iters=['mod'],
                     extra_keys=['sub_id', 'ses_id']),
        LoadImaged(keys=['query', 'key'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['query', 'key'], spatial_size=patch_size, mode='constant'),
        RandFlipd(keys=['query'], spatial_axis=(0, 1), prob=0.3),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['query'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=patch_size),
        RandSpatialCropd(keys=['key'], roi_size=patch_size),
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

@register(RK.TRANSFORM)
def get_contrastive_val_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
) -> list[Callable]:
    return [
        GetKeyQueryd(keys_prefix='img', count_key='count', extra_iters=['mod'],
                     extra_keys=['sub_id', 'ses_id']),
        LoadImaged(keys=['query', 'key'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['query', 'key'], spatial_size=patch_size, mode='constant'),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=patch_size),
        RandSpatialCropd(keys=['key'], roi_size=patch_size),
        RandScaleIntensityFixedMeand(keys=['key'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['key'], std=0.01, prob=0.2),
        RandGaussianSmoothd(keys=['key'], sigma_x=(0.5, 1.0), prob=0.2),
        RandBiasFieldd(keys=['key'], coeff_range=(0.0, 0.05), prob=0.3),
        RandGibbsNoised(keys=['key'], alpha=(0.2, 0.4), prob=0.2),
        RandAdjustContrastd(keys=['key'], gamma=(0.9, 1.1), prob=0.3),
        RandSimulateLowResolutiond(keys=['key'], prob=0.1, zoom_range=(0.5, 1.0)),
    ]

@register(RK.TRANSFORM)
def get_predict_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128), 
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True,
                   allow_missing_keys=allow_missing_keys),
        SpatialPadd(keys=keys, spatial_size=patch_size, mode='constant',
                    allow_missing_keys=allow_missing_keys),
    ]
    if concat_img:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
            ConcatItemsd(keys=keys, name='img', dim=1,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_classification_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
        RandFlipd(keys=keys, spatial_axis=(0, 1), prob=0.3, 
                  allow_missing_keys=allow_missing_keys),
        RandAffined(keys=keys, rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='zeros', prob=1.0,
                    allow_missing_keys=allow_missing_keys),
    ]
    if concat_img:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
            ConcatItemsd(keys=keys, name="img", dim=1, allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys),  # now only 'img' remains
        ])
        img_keys = ["img"]
    else:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant",
                        allow_missing_keys=allow_missing_keys),
            RandSpatialCropd(keys=keys, roi_size=patch_size,
                             allow_missing_keys=allow_missing_keys),
        ])
        img_keys = keys
        
    # intensity/artefacts AFTER shapes are aligned
    transforms.extend([
        RandScaleIntensityFixedMeand(keys=img_keys, factors=0.1, prob=0.8, allow_missing_keys=True),
        RandGaussianNoised(keys=img_keys, std=0.01, prob=0.2, allow_missing_keys=allow_missing_keys),
        RandGaussianSmoothd(keys=img_keys, sigma_x=(0.5, 1.0), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandBiasFieldd(keys=img_keys, coeff_range=(0.0, 0.05), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandGibbsNoised(keys=img_keys, alpha=(0.2, 0.4), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandAdjustContrastd(keys=img_keys, gamma=(0.9, 1.1), prob=0.5, allow_missing_keys=allow_missing_keys),
        RandSimulateLowResolutiond(keys=img_keys, prob=0.2, zoom_range=(0.7, 1.0),
                                   allow_missing_keys=allow_missing_keys),
    ])
    return transforms

@register(RK.TRANSFORM)
def get_regression_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader="NumpyReader", ensure_channel_first=True,
                   allow_missing_keys=allow_missing_keys),
        RandAffined(keys=keys, rotate_range=(0.12, 0.12, 0.12),
                    mode="bilinear", padding_mode="border", prob=0.5,
                    allow_missing_keys=allow_missing_keys),
    ]
    if concat_img:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
            ConcatItemsd(keys=keys, name="img", dim=1, allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys),  # now only 'img' remains
        ])
        img_keys = ["img"]
    else:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant",
                        allow_missing_keys=allow_missing_keys),
            RandSpatialCropd(keys=keys, roi_size=patch_size,
                             allow_missing_keys=allow_missing_keys),
        ])
        img_keys = keys

    # intensity/artefacts AFTER shapes are aligned
    transforms.extend([
        RandScaleIntensityFixedMeand(keys=img_keys, factors=0.1, prob=0.8, allow_missing_keys=True),
        RandGaussianNoised(keys=img_keys, std=0.01, prob=0.2, allow_missing_keys=allow_missing_keys),
        RandGaussianSmoothd(keys=img_keys, sigma_x=(0.5, 1.0), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandBiasFieldd(keys=img_keys, coeff_range=(0.0, 0.05), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandGibbsNoised(keys=img_keys, alpha=(0.2, 0.4), prob=0.2, allow_missing_keys=allow_missing_keys),
        RandAdjustContrastd(keys=img_keys, gamma=(0.9, 1.1), prob=0.5, allow_missing_keys=allow_missing_keys),
        RandSimulateLowResolutiond(keys=img_keys, prob=0.2, zoom_range=(0.7, 1.0),
                                   allow_missing_keys=allow_missing_keys),
    ])
    return transforms

@register(RK.TRANSFORM)
def get_classification_val_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
    ]
    if not concat_img:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode='constant', 
                        allow_missing_keys=allow_missing_keys),
            RandSpatialCropd(keys=keys, roi_size=patch_size, 
                             allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
            ConcatItemsd(keys=keys, name='img', dim=1,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys)
        ])
    return transforms