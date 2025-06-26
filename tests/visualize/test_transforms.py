"""
Visualize transforms.
"""

import pytest

from anyBrainer.transforms import (
    get_mae_train_transforms, 
    get_contrastive_train_transforms,
)
from anyBrainer.transforms import DeterministicCompose
from .utils import visualize_transform_stage

mae_settings = {
    'data': {
        "img": 'tests/examples/t1.npy', 
        "brain_mask": 'tests/examples/mask.npy',
    },
    'settings': {
        'transforms': get_mae_train_transforms(),
        'keys': ['img', 'mask', 'recon', 'brain_mask'],
        'stage': None,
        'master_seed': 340,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }
}

contrastive_settings = {
    'data': {
        "key": 'tests/examples/t1.npy',
        "query": 'tests/examples/t1.npy',
    }, 
    'settings': {
        'transforms': get_contrastive_train_transforms(),
        'keys': ['key', 'query'],
        'stage': None,
        'master_seed': 340,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }
}


@pytest.mark.viz
def test_visualize_mae_transforms():
    """Visualize the reference transforms"""
    transforms = DeterministicCompose(mae_settings['settings']['transforms'], 
                                      master_seed=mae_settings['settings']['master_seed'])
    visualize_transform_stage(
        pipeline=transforms, 
        sample=mae_settings['data'], 
        keys=mae_settings['settings']['keys'], 
        stage=mae_settings['settings']['stage'], 
        slice_indices=mae_settings['settings']['slice_indices'], 
        axis=mae_settings['settings']['axis'],
        channel=mae_settings['settings']['channel'],
        save_path=mae_settings['settings']['save_path'])

@pytest.mark.viz
def test_visualize_contrastive_transforms():
    """Visualize the reference transforms"""
    transforms = DeterministicCompose(contrastive_settings['settings']['transforms'], 
                                      master_seed=contrastive_settings['settings']['master_seed'])
    visualize_transform_stage(
        pipeline=transforms, 
        sample=contrastive_settings['data'], 
        keys=contrastive_settings['settings']['keys'], 
        stage=contrastive_settings['settings']['stage'], 
        slice_indices=contrastive_settings['settings']['slice_indices'], 
        axis=contrastive_settings['settings']['axis'],
        channel=contrastive_settings['settings']['channel'],
        save_path=contrastive_settings['settings']['save_path'])
