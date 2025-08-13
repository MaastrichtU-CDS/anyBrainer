"""
Contains functions that return lists of MONAI transforms for 
different tasks. 
"""

__all__ = [
    'get_mae_train_transforms',
    'get_mae_val_transforms',
    'get_contrastive_train_transforms',
    'get_contrastive_val_transforms',
    'get_predict_transforms',
    'get_classification_train_transforms',
    'get_segmentation_train_transforms',
    'get_downstream_val_transforms',
]

from typing import Sequence, Callable, Any
import logging

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    OneOf,
    DeleteItemsd,
    ConcatItemsd,
    LoadImaged,
    SpatialPadd,
    RandFlipd, 
    RandAffined,
    Rand3DElasticd,
    RandSpatialCropd, 
    RandSimulateLowResolutiond,
    RandScaleIntensityFixedMeand, 
    RandRicianNoised,
    RandGaussianSmoothd,
    RandBiasFieldd, 
    RandGibbsNoised, 
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    Activations,
    AsDiscrete,
    RemoveSmallObjects,
    KeepLargestConnectedComponent,
    EnsureType,
    Flip, 
)
import torch

from .unit_transforms import (
    SaveReconstructionTargetd,
    CreateRandomMaskd,
    CreateEmptyMaskd,
    GetKeyQueryd,
    SlidingWindowPatchd,
    RandImgKeyd,
)

from anyBrainer.registry import register, RegistryKind as RK

OPEN_KEYS = [f"img_{i}" for i in range(0, 20)]

logger = logging.getLogger(__name__)

@register(RK.TRANSFORM)
def get_mae_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
) -> list[Callable]:
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        CreateEmptyMaskd(mask_key='brain_mask'),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=patch_size, 
                   mode='edge'),
        RandFlipd(keys=['img', 'brain_mask'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['img', 'brain_mask'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode=['bilinear', 'nearest'], padding_mode='border', prob=1.0),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=patch_size),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
        RandScaleIntensityFixedMeand(keys=['img'], factors=0.1, prob=0.3),
        RandRicianNoised(keys=['img'], std=0.01, prob=0.2),
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
                   mode='edge'),
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
        SpatialPadd(keys=['query', 'key'], spatial_size=patch_size, mode='edge'),
        RandFlipd(keys=['query'], spatial_axis=(0, 1), prob=0.3),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['query'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='border', prob=1.0),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='border', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=patch_size),
        RandSpatialCropd(keys=['key'], roi_size=patch_size),
        RandScaleIntensityFixedMeand(keys=['query'], factors=0.1, prob=0.3),
        RandScaleIntensityFixedMeand(keys=['key'], factors=0.1, prob=0.3),
        RandRicianNoised(keys=['query'], std=0.01, prob=0.2),
        RandRicianNoised(keys=['key'], std=0.01, prob=0.2),
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
        SpatialPadd(keys=['query', 'key'], spatial_size=patch_size, mode='edge'),
        RandFlipd(keys=['key'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['key'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode='bilinear', padding_mode='border', prob=1.0),
        RandSpatialCropd(keys=['query'], roi_size=patch_size),
        RandSpatialCropd(keys=['key'], roi_size=patch_size),
        RandScaleIntensityFixedMeand(keys=['key'], factors=0.1, prob=0.3),
        RandRicianNoised(keys=['key'], std=0.01, prob=0.2),
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
        SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge',
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
    """
    IO transforms + augmentations for classification tasks.
    """
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
        RandFlipd(keys=keys, spatial_axis=(0, 1), prob=0.3, 
                  allow_missing_keys=allow_missing_keys),
        RandAffined(keys=keys, rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.1, 0.1, 0.1),
                    mode='bilinear', padding_mode='border', prob=0.5,
                    allow_missing_keys=allow_missing_keys),
    ]
    for key in keys: # unique intensity augmentations for each modality
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                         allow_missing_keys=allow_missing_keys),
            RandRicianNoised(keys=key, std=0.01, prob=0.3, 
                               allow_missing_keys=allow_missing_keys),
        ])
        # Simulate artefacts
        transforms.extend([
            OneOf(transforms=[
                RandGaussianSmoothd(keys=key, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                    sigma_z=(0.5, 1.0), prob=0.7, 
                                    allow_missing_keys=allow_missing_keys),
                RandBiasFieldd(keys=key, coeff_range=(0.0, 0.05), prob=0.7, 
                               allow_missing_keys=allow_missing_keys),
                RandGibbsNoised(keys=key, alpha=(0.2, 0.4), prob=0.7, 
                                allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0, 1.0]),
        ])
        # Simulate different acquisitions
        transforms.extend([
            OneOf(transforms=[
                RandAdjustContrastd(keys=key, gamma=(0.9, 1.1), prob=1.0, 
                                    allow_missing_keys=allow_missing_keys),
                RandSimulateLowResolutiond(keys=key, prob=0.5, zoom_range=(0.8, 1.0),
                                           allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0]),
        ])
    # Finalize
    if not concat_img:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge', 
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

@register(RK.TRANSFORM)
def get_regression_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms + augmentations for regression tasks.
    """
    transforms: list[Callable] = [
        LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
        RandAffined(keys=keys, rotate_range=(0.1, 0.1, 0.1),
                    mode='bilinear', padding_mode='border', prob=0.5,
                    allow_missing_keys=allow_missing_keys),
    ]
    for key in keys: # unique intensity augmentations for each modality
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                         allow_missing_keys=allow_missing_keys),
            RandRicianNoised(keys=key, std=0.01, prob=0.3, 
                               allow_missing_keys=allow_missing_keys),
        ])
        # Simulate artefacts
        transforms.extend([
            OneOf(transforms=[
                RandGaussianSmoothd(keys=key, sigma_x=(0.5, 1.0), prob=0.7, 
                                    allow_missing_keys=allow_missing_keys),
                RandBiasFieldd(keys=key, coeff_range=(0.0, 0.05), prob=0.7, 
                               allow_missing_keys=allow_missing_keys),
                RandGibbsNoised(keys=key, alpha=(0.2, 0.4), prob=0.7, 
                                allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0, 1.0]),
        ])
        # Simulate different acquisitions
        transforms.extend([
            OneOf(transforms=[
                RandAdjustContrastd(keys=key, gamma=(0.9, 1.1), prob=1.0, 
                                    allow_missing_keys=allow_missing_keys),
                RandSimulateLowResolutiond(keys=key, prob=0.5, zoom_range=(0.9, 1.0),
                                           allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0]),
        ])
    # Finalize
    if not concat_img:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge', 
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

@register(RK.TRANSFORM)
def get_segmentation_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    choose_one_of: bool = False,
    seg_key: str = "seg",
    allow_missing_keys: bool = True,
    concat_img: bool = False,
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms + augmentations for segmentation tasks.
    """
    all_keys = [seg_key, *keys]
    img_keys = keys
    pad_mode = ['zeros'] + ['border'] * len(keys)
    interp_mode = ['nearest'] + ['bilinear'] * len(keys)
    transforms: list[Callable] = []

    if choose_one_of:
        if concat_img:
            logger.warning("`concat_img` is ignored when `choose_one_of` is True")
        transforms.extend([
            RandImgKeyd(keys=keys, new_key='img', allow_missing_keys=allow_missing_keys),
        ])
        img_keys = ['img']
        all_keys = [seg_key, 'img']

    transforms.extend([
        LoadImaged(keys=all_keys, reader='NumpyReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
        RandFlipd(keys=all_keys, spatial_axis=(0, 1), prob=0.3, 
                  allow_missing_keys=allow_missing_keys),
        RandAffined(keys=all_keys, rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1), mode=interp_mode, 
                    padding_mode=pad_mode, prob=1.0,
                    allow_missing_keys=allow_missing_keys),
        Rand3DElasticd(keys=all_keys, sigma_range=(4, 8), prob=0.2,
                       magnitude_range=(0.5, 1.5),
                       allow_missing_keys=allow_missing_keys),
    ])
    for key in img_keys: # unique intensity augmentations for each modality
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                         allow_missing_keys=allow_missing_keys),
            RandRicianNoised(keys=key, std=0.01, prob=0.3, 
                               allow_missing_keys=allow_missing_keys),
        ])
        # Simulate artefacts
        transforms.extend([
            OneOf(transforms=[
                RandGaussianSmoothd(keys=key, sigma_x=(0.5, 1.0), prob=0.7, 
                                    allow_missing_keys=allow_missing_keys),
                RandBiasFieldd(keys=key, coeff_range=(0.0, 0.05), prob=0.7, 
                               allow_missing_keys=allow_missing_keys),
                RandGibbsNoised(keys=key, alpha=(0.2, 0.4), prob=0.7, 
                                allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0, 1.0]),
        ])
        # Simulate different acquisitions
        transforms.extend([
            OneOf(transforms=[
                RandAdjustContrastd(keys=key, gamma=(0.9, 1.1), prob=1.0, 
                                    allow_missing_keys=allow_missing_keys),
                RandSimulateLowResolutiond(keys=key, prob=0.5, zoom_range=(0.8, 1.0),
                                           allow_missing_keys=allow_missing_keys),
            ], weights=[1.0, 1.0]),
        ])
    # Finalize
    if not concat_img or choose_one_of:
        transforms.extend([
            SpatialPadd(keys=all_keys, spatial_size=patch_size, mode=pad_mode, 
                        allow_missing_keys=allow_missing_keys),
            RandCropByPosNegLabeld(keys=all_keys, label_key=seg_key, 
                                   spatial_size=patch_size, 
                                   pos=1, neg=2, num_samples=1,
                                   allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            SlidingWindowPatchd(keys=all_keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
            ConcatItemsd(keys=img_keys, name='img', dim=1,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=img_keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_downstream_val_transforms(
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
            SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge', 
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

@register(RK.TRANSFORM)
def get_postprocess_classification_transforms(
    activations_kwargs: dict[str, Any] | None = None,
    as_discrete_kwargs: dict[str, Any] | None = None,
) -> list[Callable]:
    if activations_kwargs is None:
        activations_kwargs = {'sigmoid': True}
    if as_discrete_kwargs is None:
        as_discrete_kwargs = {'threshold': 0.5}
    return [
        Activations(**activations_kwargs),
        AsDiscrete(**as_discrete_kwargs),
        EnsureType()
    ]

@register(RK.TRANSFORM)
def get_postprocess_segmentation_transforms(
    min_size: int = 1,
    threshold: float = 0.5,
) -> list[Callable]:    
    return [
        Activations(sigmoid=True),
        AsDiscrete(threshold=threshold),
        RemoveSmallObjects(min_size=min_size),
        KeepLargestConnectedComponent(applied_labels=[1]),
        EnsureType()
    ]

@register(RK.TRANSFORM)
def get_segmentation_tta() -> list[Compose]:
    """
    Get test time augmentation (TTA) transforms for segmentation tasks.

    Contains only array-based transforms; inputs will be processed as torch.Tensor
    and converted to monai.data.MetaTensor to track metadata. 

    They are returned as a list of `Compose` objects, each containing the
    transforms for a single TTA run.
    """
    tta_list: list[list[Callable]] = []
    all_flip_axis = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    for flip_axis in all_flip_axis:
        tta_list.append([
            Flip(spatial_axis=flip_axis),
        ])
    return [Compose(tta) for tta in tta_list]
