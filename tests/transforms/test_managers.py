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
    DeterministicCompose,
)

from anyBrainer.transforms.managers import (
    MAETransformManager,
    ContrastiveTransformManager,
)

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

@pytest.fixture(scope="module")
def sample_data():
    img, seg = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "img": torch.tensor(img), 
        "img_1": torch.tensor(img),
        "brain_mask": torch.tensor(seg).long(),
        "sub_id": "1",
        "ses_id": "1",
        "modality": "t1",
        "count": 2,
    }


@pytest.fixture(scope="module")
def sample_data_contrastive():
    img, _ = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "query": torch.tensor(img),
        "key": torch.tensor(img),
        "sub_id": "1",
        "ses_id": "1",
        "modality": "t1",
        "count": 2,
    }


@pytest.fixture(scope="module")
def sample_config():
    return {
        'patch_size': (128, 128, 128),
        'load_transforms': {
            'do_all': True,
        },
        'spatial_transforms': {
            'do_all': True,
        },
        'intensity_transforms': {
            'do_all': True,
        },
        'mae_transforms': {
            'do_all': True,
        },
        'patch_size_2': (128, 128, 128),
        'load_transforms_2': {
            'do_all': True,
        },
        'spatial_transforms_2': {
            'do_all': True,
        },
        'intensity_transforms_2': {
            'do_all': True,
        },
    }


set_determinism(seed=12345)

raw_mae_train_transforms = [
    LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', ensure_channel_first=True),
    SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), mode='constant'),
    RandFlipd(keys=['img', 'brain_mask'], spatial_axis=(0, 1), prob=0.3),
    RandAffined(keys=['img', 'brain_mask'], rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1), shear_range=(0.05, 0.05, 0.05),
                mode=['bilinear', 'nearest'], padding_mode=['zeros', 'zeros'], prob=0.3),
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

raw_mae_val_transforms = [
    LoadImaged(keys=['img', 'brain_mask'], reader='NumpyReader', ensure_channel_first=True),
    SpatialPadd(keys=['img', 'brain_mask'], spatial_size=(128, 128, 128), mode='constant'),
    RandSpatialCropd(keys=['img', 'brain_mask'], roi_size=(128, 128, 128)),
    SaveReconstructionTargetd(keys=['img'], recon_key='recon'),
    CreateRandomMaskd(keys=['img'], mask_key='mask', mask_ratio=0.6, mask_patch_size=32),
]


@pytest.mark.slow
class TestMAETransformManager:
    @pytest.fixture
    def mae_transforms(self, sample_config):
        return MAETransformManager(sample_config).get_train_transforms()

    @pytest.fixture
    def ref_transforms(self):
        return Compose(raw_mae_train_transforms)
    
    def test_compose(self, mae_transforms):
        transforms = mae_transforms
        assert isinstance(transforms, Compose)
    
    def test_output_keys(self, mae_transforms, sample_data):
        transforms = mae_transforms
        out = transforms(sample_data)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'mask', 'recon',
                                   'sub_id', 'ses_id', 'modality', 'count'}

    def test_output_types(self, mae_transforms, sample_data):
        transforms = mae_transforms
        out = transforms(sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['img_1'], torch.Tensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)
    
    def test_output_values(self, sample_data, sample_config):
        transforms = DeterministicCompose(MAETransformManager(sample_config).get_train_transforms(), master_seed=12345)
        ref_transforms = DeterministicCompose(raw_mae_train_transforms, master_seed=12345)
        out = transforms(sample_data)
        out_expected = ref_transforms(sample_data)
        
        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore
        assert (out['mask'] == out_expected['mask']).all() # type: ignore
        assert (out['recon'] == out_expected['recon']).all() # type: ignore


@pytest.mark.slow
class TestMAETransformManagerVal:
    @pytest.fixture
    def mae_val_transforms(self, sample_config):
        return MAETransformManager(sample_config).get_val_transforms()

    @pytest.fixture
    def ref_val_transforms(self):
        return Compose(raw_mae_val_transforms)

    def test_compose(self, mae_val_transforms):
        assert isinstance(mae_val_transforms, Compose)

    def test_output_keys(self, mae_val_transforms, sample_data):
        out = mae_val_transforms(sample_data)
        assert set(out.keys()) == {"img", "img_1", "brain_mask", "mask", "recon",
                                   "sub_id", "ses_id", "modality", "count"}

    def test_output_types(self, mae_val_transforms, sample_data):
        out = mae_val_transforms(sample_data)
        assert isinstance(out["img"], MetaTensor)
        assert isinstance(out["img_1"], torch.Tensor)
        assert isinstance(out["brain_mask"], MetaTensor)
        assert isinstance(out["mask"], MetaTensor)

    def test_output_values(self, sample_config, sample_data):
        transforms = DeterministicCompose(MAETransformManager(sample_config).get_val_transforms(), master_seed=12345)
        ref_transforms = DeterministicCompose(raw_mae_val_transforms, master_seed=12345)
        out = transforms(sample_data)
        out_expected = ref_transforms(sample_data)
        assert (out["img"] == out_expected["img"]).all()  # type: ignore
        assert (out["brain_mask"] == out_expected["brain_mask"]).all()  # type: ignore
        assert (out["mask"] == out_expected["mask"]).all()  # type: ignore
        assert (out["recon"] == out_expected["recon"]).all()  # type: ignore