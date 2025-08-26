"""
Visualize transforms.
"""

import pytest

from anyBrainer.core.transforms import (
    get_mae_train_transforms, 
    get_contrastive_train_transforms,
    get_predict_transforms,
    get_classification_train_transforms,
)
# pyright: reportPrivateImportUsage=false
from monai.transforms import Compose

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
        'master_seed': 12345,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }
}

contrastive_settings = {
    'data': {
        "img_0": 'tests/examples/t1.npy',
        "img_1": 'tests/examples/t1.npy',
        "mod_0": 't1',
        "mod_1": 't1',
        "sub_id": 0,
        "ses_id": 0,
        "count": 2
    }, 
    'settings': {
        'transforms': get_contrastive_train_transforms(),
        'keys': ['query', 'key'],
        'stage': None,
        'master_seed': 40,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }
}

predict_settings = {
    'data': {
        "flair": 'tests/examples/flair.nii.gz',
    },
    'settings': {
        'transforms': get_predict_transforms(
            patch_size=(128, 128, 128),
            keys=['flair'],
            allow_missing_keys=False,
            is_nifti=True,
            concat_img=False,
            sliding_window=False,
        ),
        'keys': ['flair'],
        'stage': None,
        'master_seed': 12,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }
}

@pytest.mark.viz
def test_visualize_mae_transforms():
    """Visualize the reference transforms"""
    transforms = Compose(
        mae_settings['settings']['transforms']
    ).set_random_state(seed=mae_settings['settings']['master_seed'])

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
    transforms = Compose(
        contrastive_settings['settings']['transforms']
    ).set_random_state(seed=contrastive_settings['settings']['master_seed'])
    
    visualize_transform_stage(
        pipeline=transforms, 
        sample=contrastive_settings['data'], 
        keys=contrastive_settings['settings']['keys'], 
        stage=contrastive_settings['settings']['stage'], 
        slice_indices=contrastive_settings['settings']['slice_indices'], 
        axis=contrastive_settings['settings']['axis'],
        channel=contrastive_settings['settings']['channel'],
        save_path=contrastive_settings['settings']['save_path'])

@pytest.mark.viz
def test_visualize_predict_transforms():
    """Visualize the reference transforms"""
    transforms = Compose(
        predict_settings['settings']['transforms']
    ).set_random_state(seed=predict_settings['settings']['master_seed'])
    
    visualize_transform_stage(
        pipeline=transforms,
        sample=predict_settings['data'],
        keys=predict_settings['settings']['keys'],
        stage=predict_settings['settings']['stage'],
        slice_indices=predict_settings['settings']['slice_indices'],
        axis=predict_settings['settings']['axis'],
        channel=predict_settings['settings']['channel'],
        save_path=predict_settings['settings']['save_path'])