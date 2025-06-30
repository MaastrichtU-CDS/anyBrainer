"""Tests for transform managers"""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d
from monai.utils import set_determinism
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImage,
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

from anyBrainer.transforms import (
    SaveReconstructionTargetd,
    CreateRandomMaskd,
)
from anyBrainer.transforms import (
    get_mae_train_transforms,
    get_mae_val_transforms,
    get_contrastive_train_transforms,
    get_contrastive_val_transforms,
)
from anyBrainer.transforms import DeterministicCompose

set_determinism(seed=12345)

@pytest.fixture(autouse=True)
def mock_load_image(monkeypatch):
    """
    Monkey-patch LoadImage so every attempt to read a file
    yields a synthetic 3-D volume instead of touching the disk.
    """
    def _dummy_call(self, filename, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(hash(filename) & 0xFFFF_FFFF)
        img = torch.rand((1, 120, 120, 120), dtype=torch.float32, generator=gen)
        # LoadImage normally returns (np.ndarray, meta_dict)
        return img

    monkeypatch.setattr(LoadImage, "__call__", _dummy_call, raising=True)

ref_mae_train_transforms = [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), mode='constant'),
        RandFlipd(keys=['img', 'brain_mask'], spatial_axis=(0, 1), prob=0.3),
        RandAffined(keys=['img', 'brain_mask'], rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                    mode=['bilinear', 'nearest'], padding_mode='zeros', prob=1.0),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128)),
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

ref_mae_val_transforms = [
        LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', ensure_channel_first=True),
        SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), mode='constant'),
        RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128)),
        SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
        CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6,
                          mask_patch_size=32),
    ]

ref_contrastive_train_transforms = [
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

ref_contrastive_val_transforms = [
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
    

@pytest.mark.slow
class TestMAETrainTransforms:
    def test_output_keys(self, sample_data):
        op = Compose(get_mae_train_transforms())
        out = op(sample_data)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'mask', 'recon', # type: ignore
                                   'sub_id', 'ses_id', 'modality', 'count'} # type: ignore
    
    def test_output_types(self, sample_data):
        op = Compose(get_mae_train_transforms())
        out = op(sample_data)
        assert isinstance(out['img'], MetaTensor) # type: ignore
        assert isinstance(out['img_1'], torch.Tensor) # type: ignore        
        assert isinstance(out['brain_mask'], MetaTensor) # type: ignore
        assert isinstance(out['mask'], MetaTensor) # type: ignore
        assert isinstance(out['recon'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['modality'], str) # type: ignore
        assert isinstance(out['count'], int) # type: ignore
    
    def test_output_values(self, sample_data):
        t1 = Compose(get_mae_train_transforms())
        t2 = Compose(ref_mae_train_transforms)
        out = t1(sample_data)
        out_expected = t2(sample_data)
        
        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore
        assert (out['mask'] == out_expected['mask']).all() # type: ignore
        assert (out['recon'] == out_expected['recon']).all() # type: ignore

@pytest.mark.slow
class TestMAEValTransforms:
    def test_output_keys(self, sample_data):
        op = Compose(get_mae_val_transforms())
        out = op(sample_data)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'mask', 'recon', # type: ignore
                                   'sub_id', 'ses_id', 'modality', 'count'} # type: ignore
    
    def test_output_types(self, sample_data):
        op = Compose(get_mae_val_transforms())
        out = op(sample_data)
        assert isinstance(out['img'], MetaTensor) # type: ignore
        assert isinstance(out['img_1'], torch.Tensor) # type: ignore        
        assert isinstance(out['brain_mask'], MetaTensor) # type: ignore
        assert isinstance(out['mask'], MetaTensor) # type: ignore
        assert isinstance(out['recon'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['modality'], str) # type: ignore
        assert isinstance(out['count'], int) # type: ignore
    
    def test_output_values(self, sample_data):
        t1 = Compose(get_mae_val_transforms())
        t2 = Compose(ref_mae_val_transforms)
        out = t1(sample_data)
        out_expected = t2(sample_data)
        
        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore
        assert (out['mask'] == out_expected['mask']).all() # type: ignore
        assert (out['recon'] == out_expected['recon']).all() # type: ignore



@pytest.mark.slow
class TestContrastiveTrainTransforms:
    def test_output_keys(self, sample_data_contrastive):
        op = Compose(get_contrastive_train_transforms())
        out = op(sample_data_contrastive)
        assert set(out.keys()) == {'query', 'key', 'sub_id', 'ses_id', 'modality', 'count'} # type: ignore

    def test_output_types(self, sample_data_contrastive):
        op = Compose(get_contrastive_train_transforms())
        out = op(sample_data_contrastive)
        assert isinstance(out['query'], MetaTensor) # type: ignore
        assert isinstance(out['key'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['modality'], str) # type: ignore
        assert isinstance(out['count'], int) # type: ignore
    
    def test_output_values(self, sample_data_contrastive):
        t1 = Compose(get_contrastive_train_transforms())
        t2 = Compose(ref_contrastive_train_transforms)
        out = t1(sample_data_contrastive)
        out_expected = t2(sample_data_contrastive)
        
        assert (out['query'] == out_expected['query']).all() # type: ignore
        assert (out['key'] == out_expected['key']).all() # type: ignore
    
@pytest.mark.slow
class TestContrastiveValTransforms:
    def test_output_keys(self, sample_data_contrastive):
        op = Compose(get_contrastive_val_transforms())
        out = op(sample_data_contrastive)
        assert set(out.keys()) == {'query', 'key', 'sub_id', 'ses_id', 'modality', 'count'} # type: ignore

    def test_output_types(self, sample_data_contrastive):
        op = Compose(get_contrastive_val_transforms())
        out = op(sample_data_contrastive)
        assert isinstance(out['query'], MetaTensor) # type: ignore
        assert isinstance(out['key'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['modality'], str) # type: ignore
        assert isinstance(out['count'], int) # type: ignore
    
    def test_output_values(self, sample_data_contrastive):
        t1 = Compose(get_contrastive_val_transforms())
        t2 = Compose(ref_contrastive_val_transforms)
        out = t1(sample_data_contrastive)
        out_expected = t2(sample_data_contrastive)
        
        assert (out['query'] == out_expected['query']).all() # type: ignore
        assert (out['key'] == out_expected['key']).all() # type: ignore