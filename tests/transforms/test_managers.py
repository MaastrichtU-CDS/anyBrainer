"""Tests for transform managers"""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d
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

@pytest.fixture(scope="module")
def sample_config():
    return {
        'patch_size': (128, 128, 128),
        'load_transforms': {
            'load': { # Should see the transform despite missing 'enabled'
                'enabled': True,
                'params': {
                    'reader': None, # should replace 'NumpyReader' with MONAI's default
                    'ensure_channel_first': True,
                }
            },
            'pad': {
                'enabled': True,
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
                    'mask_patch_size': 6,
                }
            },
        },
    }

class TestMAETransformManager:
    @pytest.fixture
    def manager(self, sample_config):
        config = {'patch_size': sample_config['patch_size']}
        config.update(**sample_config['mae_transforms'])
        return MAETransformManager(config)