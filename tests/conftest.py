import sys
from pathlib import Path
import logging
import torch.multiprocessing as mp

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
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
from monai.data import create_test_image_3d

from anyBrainer.transforms import (
    SaveReconstructionTargetd, 
    CreateRandomMaskd,
    CreateEmptyMaskd,
    GetKeyQueryd,
)

logger = logging.getLogger("anyBrainer")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mp.set_start_method("fork", force=True) # Replace with worker_init_fn when seeding ready.

if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False


@pytest.fixture(scope="session")
def mae_sample_data():
    img, seg = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "img": torch.tensor(img), 
        "brain_mask": torch.tensor(seg).long(),
        "sub_id": "1",
        "ses_id": "1",
        "mod": "t1",
    }

@pytest.fixture(scope="session")
def contrastive_sample_data():
    return {
        "img_0": create_test_image_3d(120, 120, 120, channel_dim=0)[0], 
        "img_1": create_test_image_3d(120, 120, 120, channel_dim=0)[0],
        "img_2": create_test_image_3d(120, 120, 120, channel_dim=0)[0],
        "mod_0": "t1",
        "mod_1": "t2",
        "mod_2": "flair",
        "sub_id": "1",
        "ses_id": "1",
        "count": 3,
    }

@pytest.fixture(scope="session")
def ref_mae_train_transforms():
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', 
                   ensure_channel_first=True, allow_missing_keys=True),
        CreateEmptyMaskd(mask_key='brain_mask'),
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


@pytest.fixture(scope="session")
def ref_mae_val_transforms():
    return [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', ensure_channel_first=True),
        CreateEmptyMaskd(mask_key='brain_mask'),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), mode='constant'),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128)),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
    ]

@pytest.fixture(scope="session")
def ref_contrastive_train_transforms():
    return [
        GetKeyQueryd(keys_prefix='img', count_key='count', extra_iters=['mod'],
                     extra_keys=['sub_id', 'ses_id']),
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

@pytest.fixture(scope="session")
def ref_contrastive_val_transforms():
    return [
        GetKeyQueryd(keys_prefix='img', count_key='count', extra_iters=['mod'],
                     extra_keys=['sub_id', 'ses_id']),
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

