"""Tests for transform managers"""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d
from monai.utils import set_determinism
from monai.data import MetaTensor
from monai.transforms import (
    LoadImage,
    LoadImaged,
    SpatialPadd,
    Compose,
)

from anyBrainer.transforms.managers import (
    MAETransformManager,
    ContrastiveTransformManager,
)
from anyBrainer.transforms.builders import (
    LoadTransformBuilder,
    SpatialTransformBuilder,
    IntensityTransformBuilder,
    MaskingTransformBuilder,
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
class TestMAETransformManager:
    @pytest.fixture
    def manager(self, sample_config):
        return MAETransformManager(sample_config)
    
    def test_compose(self, manager):
        transforms = manager.get_train_transforms()
        assert isinstance(transforms, Compose)
    
    def test_output_keys(self, manager, sample_data):
        transforms = manager.get_train_transforms()
        out = transforms(sample_data)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'mask', 'recon',
                                   'sub_id', 'ses_id', 'modality', 'count'}

    def test_output_types(self, manager, sample_data):
        transforms = manager.get_train_transforms()
        out = transforms(sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['img_1'], torch.Tensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)
    
    def test_output_values(self, manager, sample_data, sample_config):
        transforms = manager.get_train_transforms()
        out = transforms(sample_data)

        config = sample_config.copy()

        load_builder = LoadTransformBuilder(config['load_transforms'])
        spatial_builder = SpatialTransformBuilder(config['spatial_transforms'])
        intensity_builder = IntensityTransformBuilder(config['intensity_transforms'])
        masking_builder = MaskingTransformBuilder(config['mae_transforms'])

        expected_load_transforms = load_builder.build(img_keys=["img", "brain_mask"])
        expected_spatial_transforms = spatial_builder.build(img_keys=["img", "brain_mask"])
        expected_masking_transforms = masking_builder.build(img_keys=["img"])
        expected_intensity_transforms = intensity_builder.build(img_keys=["img"])

        expected_transforms = Compose([
            expected_load_transforms,
            expected_spatial_transforms,
            expected_masking_transforms,
            expected_intensity_transforms,
        ])
        out_expected = expected_transforms(sample_data)

        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore
        assert (out['mask'] == out_expected['mask']).all() # type: ignore
       
    def test_reconstruction_target(self, manager, sample_data, sample_config):
        transforms = manager.get_train_transforms()
        out = transforms(sample_data)

        config = sample_config.copy()
        load_builder = LoadTransformBuilder(config['load_transforms'])
        spatial_builder = SpatialTransformBuilder(config['spatial_transforms'])

        expected_load_transforms = load_builder.build(img_keys=["img", "brain_mask"])
        expected_spatial_transforms = spatial_builder.build(img_keys=["img", "brain_mask"])
        
        till_spatial_transforms = Compose([
            expected_load_transforms,
            expected_spatial_transforms,
        ])
        spatial_output = till_spatial_transforms(sample_data)
        assert (out['recon'] == spatial_output['img']).all() # type: ignore
    