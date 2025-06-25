"""Tests for transform builders"""

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
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
)

from anyBrainer.transforms import (
    CreateRandomMaskd,
)

from anyBrainer.transforms.builders import (
    LoadTransformBuilder,
    SpatialTransformBuilder,
    IntensityTransformBuilder,
    MaskingTransformBuilder,
)

set_determinism(seed=12345)

@pytest.fixture(scope="module")
def sample_config():
    return {
        'patch_size': (128, 128, 128),
        'load_transforms': {
            'do_all': True,
            'load': { # Should see the transform despite missing 'enabled'
                'enabled': True,
                'params': {
                    'reader': None, # should replace 'NumpyReader' with MONAI's default
                    'ensure_channel_first': True,
                }
            },
            'pad': {
                'enabled': False, # should still do, since do_all is True
                'mode': 'reflect', # should see 'constant' since not in 'params'
            },
        },
        'spatial_transforms': {
            'flip': { # Should see this transform with defaults
                'enabled': True,
            }, 
            'affine': { # Should see this transform
                'enabled': True,
                'params': {
                    'mode': ['bilinear', 'bilinear', 'bilinear'], # should see it as list instead of str
                    'padding_mode': 'reflect' # should remain str
                }
            } 
        },
        'intensity_transforms': {
            'scale_intensity': {
                'enabled': False,
            },
            'gaussian_noise': { # should see only this, default settings
                'enabled': True,
            },
        },
        'mae_transforms': {
            'reconstruction_target': {
                'enabled': False,
            },
            'random_mask': { # should see this transform
                'enabled': True,
                'params': {
                    'mask_ratio': 0.3, # should replace default values
                    'mask_patch_size': 16,
                }
            },
        },
    }

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


class TestLoadTransformBuilder:
    expected_transform_params = {
        'load': {
                'keys': ["img", "img_1", "brain_mask"],
                'allow_missing_keys': False,
                'reader': None,
                'ensure_channel_first': True,
            },
            'pad': {
                'keys': ["img", "img_1", "brain_mask"],
                'allow_missing_keys': False,
                'spatial_size': (128, 128, 128),
                'mode': 'constant',
            },
    }

    @pytest.fixture
    def builder(self, sample_config):
        config = {'patch_size': sample_config['patch_size']}
        config.update(**sample_config['load_transforms'])
        return LoadTransformBuilder(config)
    
    def test_compose(self, builder):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert isinstance(transforms, Compose)
    
    def test_transform_params(self, builder):
        builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert builder.params == self.expected_transform_params
    
    def test_output_keys(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'sub_id', 
                                   'ses_id', 'modality', 'count'}

    def test_output_types(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['img_1'], MetaTensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)
    
    def test_outputs_directly(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        
        expected_transforms = Compose([
            LoadImaged(**self.expected_transform_params['load']),
            SpatialPadd(**self.expected_transform_params['pad']),
        ])
        out_expected = expected_transforms(sample_data)

        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['img_1'] == out_expected['img_1']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore


class TestSpatialTransformBuilder:
    expected_transform_params = {
        'flip': {
            'keys': ["img", "img_1", "brain_mask"],
            'allow_missing_keys': False,
            'prob': 0.3,
            'spatial_axis': [0, 1],
        },
        'affine': { 
            'keys': ["img", "img_1", "brain_mask"],
            'allow_missing_keys': False,
            'prob': 0.3,
            'rotate_range': (0.1, 0.1, 0.1),
            'scale_range': (0.1, 0.1, 0.1),
            'shear_range': (0.05, 0.05, 0.05),
            'mode': ['bilinear', 'bilinear', 'bilinear'],
            'padding_mode': 'reflect'
        }
    }

    @pytest.fixture
    def builder(self, sample_config):
        config = {'patch_size': sample_config['patch_size']}
        config.update(**sample_config['spatial_transforms'])
        return SpatialTransformBuilder(config)
    
    def test_compose(self, builder):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert isinstance(transforms, Compose)
    
    def test_transform_params(self, builder):
        builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert builder.params == self.expected_transform_params
    
    def test_output_keys(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'sub_id', 
                                   'ses_id', 'modality', 'count'}

    def test_output_types(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['img_1'], MetaTensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)
    
    def test_outputs_directly(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        
        expected_transforms = Compose([
            RandFlipd(**self.expected_transform_params['flip']),
            RandAffined(**self.expected_transform_params['affine']),
        ])
        out_expected = expected_transforms(sample_data)

        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['img_1'] == out_expected['img_1']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore


class TestIntensityTransformBuilder:
    expected_transform_params = {
        'gaussian_noise': {
            'keys': ["img", "img_1", "brain_mask"],
            'allow_missing_keys': False,
            'std': 0.01,
            'prob': 0.2,
        }
    }

    @pytest.fixture
    def builder(self, sample_config):
        config = {'patch_size': sample_config['patch_size']}
        config.update(**sample_config['intensity_transforms'])
        return IntensityTransformBuilder(config)
    
    def test_compose(self, builder):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert isinstance(transforms, Compose)
    
    def test_transform_params(self, builder):
        builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert builder.params == self.expected_transform_params
    
    def test_output_keys(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'sub_id', 
                                   'ses_id', 'modality', 'count'}

    def test_output_types(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['img_1'], MetaTensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)
    
    def test_outputs_directly(self, builder, sample_data):  
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        
        expected_transforms = Compose([
            RandGaussianNoised(**self.expected_transform_params['gaussian_noise']),
        ])
        out_expected = expected_transforms(sample_data)

        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['img_1'] == out_expected['img_1']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore


class TestMaskingTransformBuilder:
    expected_transform_params = {
        'random_mask': { 
            'keys': ["img", "img_1", "brain_mask"],
            'allow_missing_keys': False,
            'mask_key': 'mask',
            'mask_ratio': 0.3,
            'mask_patch_size': 16,
        }
    }

    @pytest.fixture
    def builder(self, sample_config):
        config = {'patch_size': sample_config['patch_size']}
        config.update(**sample_config['mae_transforms'])
        return MaskingTransformBuilder(config)
    
    def test_compose(self, builder):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert isinstance(transforms, Compose)
    
    def test_transform_params(self, builder):
        builder.build(img_keys=["img", "img_1", "brain_mask"])
        assert builder.params == self.expected_transform_params
    
    def test_output_keys(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'img', 'img_1', 'brain_mask', 'mask',
                                   'sub_id', 'ses_id', 'modality', 'count'}

    def test_output_types(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        assert isinstance(out['img'], torch.Tensor)
        assert isinstance(out['img_1'], torch.Tensor)
        assert isinstance(out['brain_mask'], torch.Tensor)
        assert isinstance(out['mask'], torch.Tensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['modality'], str)
        assert isinstance(out['count'], int)

    def test_outputs_directly(self, builder, sample_data):
        transforms = builder.build(img_keys=["img", "img_1", "brain_mask"])
        out = transforms(sample_data)
        
        expected_transforms = Compose([
            CreateRandomMaskd(**self.expected_transform_params['random_mask']),
        ])
        out_expected = expected_transforms(sample_data)

        assert (out['img'] == out_expected['img']).all() # type: ignore
        assert (out['img_1'] == out_expected['img_1']).all() # type: ignore
        assert (out['brain_mask'] == out_expected['brain_mask']).all() # type: ignore
        assert (out['mask'] == out_expected['mask']).all() # type: ignore