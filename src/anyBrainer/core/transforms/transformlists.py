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
    CropForegroundd,
    ConcatItemsd,
    Orientationd,
    NormalizeIntensityd,
    LoadImaged,
    SpatialPadd,
    RandFlipd, 
    RandAffined,
    Rand3DElasticd,
    RandSpatialCropd, 
    RandSimulateLowResolutiond,
    RandScaleIntensityFixedMeand, 
    RandRicianNoised,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandBiasFieldd, 
    RandGibbsNoised, 
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    Spacingd,
    Activations,
    AsDiscrete,
    RemoveSmallObjects,
    KeepLargestConnectedComponent,
    EnsureType,
    Flipd, 
)

from .unit_transforms import (
    SaveReconstructionTargetd,
    CreateRandomMaskd,
    CreateEmptyMaskd,
    GetKeyQueryd,
    SlidingWindowPatchd,
    RandImgKeyd,
    ClipNonzeroPercentilesd,
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
    spacing: tuple[float, float, float] = (1, 1, 1),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    delete_orig: bool = True, # when concat_img is True
    sliding_window: bool = False,
    target_key: str = "img",
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms for inference.
    """
    transforms: list[Callable] = []

    # Load data; normalize and reorient if NIfTI
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True,
                       allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
            Orientationd(keys=keys, axcodes='RAS', allow_missing_keys=allow_missing_keys),
            Spacingd(keys=keys, pixdim=spacing, allow_missing_keys=allow_missing_keys),
            CropForegroundd(keys=keys, source_key=keys[0], allow_missing_keys=allow_missing_keys),
            ClipNonzeroPercentilesd(keys=keys, lower=0.5, upper=99.9, allow_missing_keys=allow_missing_keys),
            NormalizeIntensityd(keys=keys, allow_missing_keys=allow_missing_keys, nonzero=True),
        ])

    # Match expected size
    transforms.extend([
        SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge',
                    allow_missing_keys=allow_missing_keys),
    ])
    if sliding_window:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
        ])
    if concat_img:
        transforms.extend([
            ConcatItemsd(keys=keys, name=target_key, dim=1,
                        allow_missing_keys=allow_missing_keys),
        ])
        if delete_orig:
            transforms.extend([
                DeleteItemsd(keys=keys)
            ])
    return transforms

@register(RK.TRANSFORM)
def get_classification_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    sliding_window: bool = False,
    target_key: str = "img",
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms + augmentations for classification tasks.
    """
    transforms: list[Callable] = []
    
    # Load data; normalize and reorient if NIfTI
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
            Orientationd(keys=keys, axcodes='RAS', allow_missing_keys=allow_missing_keys),
            ClipNonzeroPercentilesd(keys=keys, lower=0.5, upper=99.9, 
                       allow_missing_keys=allow_missing_keys),
            NormalizeIntensityd(keys=keys, nonzero=True, allow_missing_keys=allow_missing_keys),
        ])
    
    # Augmentations
    transforms.extend([
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5, 
                    allow_missing_keys=allow_missing_keys),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5, 
                allow_missing_keys=allow_missing_keys),
        RandAffined(keys=keys, rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.1, 0.1, 0.1),
                    mode='bilinear', padding_mode='border', prob=0.5,
                    allow_missing_keys=allow_missing_keys),
    ])
    for key in keys: # unique intensity augmentations for each modality
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                         allow_missing_keys=allow_missing_keys),
            RandGaussianNoised(keys=key, std=0.01, prob=0.3, allow_missing_keys=allow_missing_keys),
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

    # Match expected size
    if not sliding_window:
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
        ])
    if concat_img:
        ch_dim = 1 if sliding_window else 0
        transforms.extend([
            ConcatItemsd(keys=keys, name=target_key, dim=ch_dim,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_regression_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    sliding_window: bool = False,
    target_key: str = "img",
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms + augmentations for regression tasks.
    """
    transforms: list[Callable] = []

    # Load data; normalize and reorient if NIfTI
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
            Orientationd(keys=keys, axcodes='RAS', allow_missing_keys=allow_missing_keys),
            ClipNonzeroPercentilesd(keys=keys, lower=0.5, upper=99.9, 
                       allow_missing_keys=allow_missing_keys),
            NormalizeIntensityd(keys=keys, nonzero=True, allow_missing_keys=allow_missing_keys),
        ])
    
    # Augmentations
    transforms.extend([
        RandAffined(keys=keys, rotate_range=(0.1, 0.1, 0.1),
                    mode='bilinear', padding_mode='border', prob=0.5,
                    allow_missing_keys=allow_missing_keys),
    ])
    for key in keys: # unique intensity augmentations for each modality
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                         allow_missing_keys=allow_missing_keys),
            RandGaussianNoised(keys=key, std=0.01, prob=0.3, allow_missing_keys=allow_missing_keys),
        ])
        # Simulate artifacts
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
    # Match expected size
    if not sliding_window:
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
        ])
    if concat_img:
        ch_dim = 1 if sliding_window else 0
        transforms.extend([
            ConcatItemsd(keys=keys, name=target_key, dim=ch_dim,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_segmentation_train_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    seg_key: str = "seg",
    create_empty_seg: bool = True,
    allow_missing_keys: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    choose_one_of: bool = False,
    target_key: str = "img",
    sliding_window: bool = False,
    n_patches: int | Sequence[int] = 1,
    n_pos: int = 1,
    n_neg: int = 2,
    overfit: bool = False,
) -> list[Callable]:
    """
    IO transforms + augmentations for segmentation tasks.

    Args:
        patch_size: Patch size.
        keys: Keys to load.
        choose_one_of: Whether to choose one of the keys randomly.
        seg_key: Key to load the segmentation mask.
        create_empty_seg: Whether to create empty segmentation masks when missing.
        allow_missing_keys: Whether to allow missing keys.
        is_nifti: Whether the data is in NIfTI format; if True, reorient and normalize.
        concat_img: Whether to concatenate the images.
        target_key: Key to use as the target when `concat_img` or `choose_one_of` is True.
        sliding_window: Whether to stack sliding window patches along `0` dimension.
        n_patches: Number of patches to extract from image.
            - If `sliding_window` is True, this is the number of patches to extract from each image.
            - If `sliding_window` is False, this is the number of random crops from `RandCropByPosNegLabeld`.
        n_pos: Relative weight of positive crops when `concat_img` is False.
        n_neg: Relative weight of negative crops when `concat_img` is False.
        overfit: Whether to overfit.
    """
    all_keys = [seg_key] + keys.copy()
    img_keys = keys.copy()
    transforms: list[Callable] = []

    # Filter keys
    _allow_missing_keys = allow_missing_keys or create_empty_seg # till mask is created
    if choose_one_of:
        if concat_img:
            logger.warning("`concat_img` is ignored when `choose_one_of` is True")
        transforms.extend([
            RandImgKeyd(keys=keys, new_key='img', allow_missing_keys=_allow_missing_keys),
            DeleteItemsd(keys=img_keys)
        ])
        img_keys = [target_key]
        all_keys = [seg_key, target_key]
        allow_missing_keys = False
    
    pad_mode_affine = ['constant'] + ['border'] * len(img_keys)
    pad_mode_spatial = ['constant'] + ['edge'] * len(img_keys)
    interp_mode = ['nearest'] + ['bilinear'] * len(img_keys)
    
    # Load data; normalize and reorient if NIfTI
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=all_keys, reader='NumpyReader', ensure_channel_first=True, 
                       allow_missing_keys=_allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=all_keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=_allow_missing_keys),
            Orientationd(keys=all_keys, axcodes='RAS', allow_missing_keys=_allow_missing_keys),
            ClipNonzeroPercentilesd(keys=img_keys, lower=0.5, upper=99.9, 
                                    allow_missing_keys=_allow_missing_keys),
            NormalizeIntensityd(keys=img_keys, nonzero=True, allow_missing_keys=_allow_missing_keys),
        ])
    
    # Ensure seg mask exists
    if create_empty_seg:
        transforms.extend([
            CreateEmptyMaskd(keys=img_keys, mask_key=seg_key, allow_missing_keys=_allow_missing_keys),
        ])

    # Augmentations
    if not overfit:
        # Spatial
        transforms.extend([
            RandFlipd(keys=all_keys, spatial_axis=0, prob=0.5, 
                    allow_missing_keys=allow_missing_keys),
            RandFlipd(keys=all_keys, spatial_axis=1, prob=0.5, 
                    allow_missing_keys=allow_missing_keys),
            RandAffined(keys=all_keys, rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1), mode=interp_mode, 
                        padding_mode=pad_mode_affine, prob=1.0,
                        allow_missing_keys=allow_missing_keys),
            Rand3DElasticd(keys=all_keys, sigma_range=(4, 8), prob=0.2,
                        magnitude_range=(0.5, 1.5), mode=interp_mode,
                        allow_missing_keys=allow_missing_keys),
        ])
        # intensity; unique for each modality
        for key in img_keys:
            transforms.extend([
                RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                            allow_missing_keys=allow_missing_keys),
                RandGaussianNoised(keys=key, std=0.01, prob=0.3, 
                                allow_missing_keys=allow_missing_keys),
            ])
            # Simulate artifacts
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

    # Match expected size
    if not sliding_window:
        if not isinstance(n_patches, int):
            msg = f"`n_patches` must be an integer when `sliding_window` is False."
            logger.error(msg)
            raise ValueError(msg)
                
        if overfit:
            n_pos, n_neg = (1, 0)

        transforms.extend([
            SpatialPadd(keys=all_keys, spatial_size=patch_size, mode=pad_mode_spatial, 
                        allow_missing_keys=allow_missing_keys),
            RandCropByPosNegLabeld(keys=all_keys, label_key=seg_key, 
                                   spatial_size=patch_size, 
                                   pos=n_pos, neg=n_neg, num_samples=n_patches,
                                   allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            SlidingWindowPatchd(keys=all_keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
        ])
    
    if concat_img and not choose_one_of:
        ch_dim = 1 if sliding_window else 0
        transforms.extend([
            ConcatItemsd(keys=img_keys, name=target_key, dim=ch_dim,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=img_keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_downstream_val_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    sliding_window: bool = False,
    target_key: str = "img",
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms for downstream tasks.
    """
    transforms: list[Callable] = []

    # Load data; normalize and reorient if NIfTI
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=keys, reader='NumpyReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
            Orientationd(keys=keys, axcodes='RAS', allow_missing_keys=allow_missing_keys),
            NormalizeIntensityd(keys=keys, allow_missing_keys=allow_missing_keys),
        ])

    # Match expected size
    if not sliding_window:
        transforms.extend([
            SpatialPadd(keys=keys, spatial_size=patch_size, mode='edge', 
                        allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            SlidingWindowPatchd(keys=keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
        ])
    if concat_img:
        ch_dim = 1 if sliding_window else 0
        transforms.extend([
            ConcatItemsd(keys=keys, name=target_key, dim=ch_dim,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_segmentation_val_transforms(
    patch_size: int | Sequence[int] = (128, 128, 128),
    keys: list[str] = OPEN_KEYS,
    seg_key: str = "seg",
    allow_missing_keys: bool = True,
    create_empty_seg: bool = True,
    is_nifti: bool = False,
    concat_img: bool = False,
    sliding_window: bool = False,
    target_key: str = "img",
    n_patches: int | Sequence[int] = (2, 2, 2),
) -> list[Callable]:
    """
    IO transforms for segmentation tasks.

    See `get_segmentation_train_transforms` for parameter documentation.
    Here `n_samples` is used only when `sliding_window` is True.
    """
    img_keys = keys.copy()
    all_keys = [seg_key] + img_keys
    pad_mode = ['constant'] + ['edge'] * len(keys)
    transforms: list[Callable] = []

    # Load data; normalize and reorient if NIfTI
    _allow_missing_keys = allow_missing_keys or create_empty_seg
    if not is_nifti:
        transforms.extend([
            LoadImaged(keys=all_keys, reader='NumpyReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            LoadImaged(keys=all_keys, reader='NibabelReader', ensure_channel_first=True, 
                       allow_missing_keys=allow_missing_keys),
            Orientationd(keys=all_keys, axcodes='RAS', allow_missing_keys=allow_missing_keys),
            NormalizeIntensityd(keys=img_keys, allow_missing_keys=allow_missing_keys),
        ])
    
    # Ensure seg mask exists
    if create_empty_seg:
        transforms.extend([
            CreateEmptyMaskd(keys=img_keys, mask_key=seg_key, allow_missing_keys=_allow_missing_keys),
        ])
    
    # Match expected size
    if not sliding_window:
        transforms.extend([
            SpatialPadd(keys=all_keys, spatial_size=patch_size, mode=pad_mode, 
                        allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            SlidingWindowPatchd(keys=all_keys, patch_size=patch_size, overlap=None,
                                n_patches=n_patches, allow_missing_keys=allow_missing_keys),
        ])
    if concat_img:
        ch_dim = 1 if sliding_window else 0
        transforms.extend([
            ConcatItemsd(keys=img_keys, name=target_key, dim=ch_dim,
                        allow_missing_keys=allow_missing_keys),
            DeleteItemsd(keys=img_keys)
        ])
    return transforms

@register(RK.TRANSFORM)
def get_postprocess_classification_transforms(
    activation_kwargs: dict[str, Any] | None = None,
    as_discrete_kwargs: dict[str, Any] | None = None,
) -> list[Callable]:
    """
    Postprocess classification outputs.

    `activations_kwargs` and `as_discrete_kwargs` are passed to the respective transforms.
    """
    if activation_kwargs is None:
        activation_kwargs = {'sigmoid': True}
    if as_discrete_kwargs is None:
        as_discrete_kwargs = {'threshold': 0.5}
    return [
        Activations(**activation_kwargs),
        AsDiscrete(**as_discrete_kwargs),
        EnsureType()
    ]

@register(RK.TRANSFORM)
def get_postprocess_segmentation_transforms(
    activation_kwargs: dict[str, Any] | None = None,
    as_discrete_kwargs: dict[str, Any] | None = None,
    keep_largest: bool = True,
    keep_largest_kwargs: dict[str, Any] | None = None,
    remove_small_objects: bool = True,
    remove_small_objects_kwargs: dict[str, Any] | None = None,
) -> list[Callable]:
    """
    Postprocess segmentation outputs.

    If `keep_largest` is True, the largest connected component is kept.
    If `remove_small_objects` is True, small objects are removed.

    `keep_largest_kwargs` and `remove_small_objects_kwargs` are passed to the respective transforms.
    """
    activation_kwargs = activation_kwargs or {'sigmoid': True}
    as_discrete_kwargs = as_discrete_kwargs or {'threshold': 0.5}
    keep_largest_kwargs = keep_largest_kwargs or {'applied_labels': [1]}
    remove_small_objects_kwargs = remove_small_objects_kwargs or {'min_size': 1}
    transforms = [
        Activations(**activation_kwargs),
        AsDiscrete(**as_discrete_kwargs),
    ]
    if keep_largest:
        transforms.append(KeepLargestConnectedComponent(**keep_largest_kwargs))
    if remove_small_objects:
        transforms.append(RemoveSmallObjects(**remove_small_objects_kwargs))
    return transforms

@register(RK.TRANSFORM)
def get_flip_tta(
    keys: list[str] = OPEN_KEYS,
    allow_missing_keys: bool = True,
) -> list[Callable]:
    """
    Get test time augmentation (TTA) transforms for segmentation tasks.

    Contains only array-based transforms; inputs will be processed as torch.Tensor
    and converted to monai.data.MetaTensor to track metadata. 

    They are returned as a list of `Compose` objects, each containing the transforms 
    for a single TTA run.
    """
    tta_list = []
    all_flip_axis = ((), (0,), (1,), (0, 1))
    for flip_axis in all_flip_axis:
        tta_list.append([
            Flipd(keys=keys, spatial_axis=flip_axis, allow_missing_keys=allow_missing_keys),
        ])
    return [Compose(tta) for tta in tta_list]