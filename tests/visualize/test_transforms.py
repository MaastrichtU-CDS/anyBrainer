"""
Visualize transforms.
"""

import pytest

from anyBrainer.transforms import get_mae_train_transforms
from anyBrainer.transforms import DeterministicCompose
from .utils import visualize_transform_stage


sample_data = {
    "img": 'tests/examples/t1.npy', 
    "brain_mask": 'tests/examples/mask.npy',
}

viz_settings = {
        'transforms': get_mae_train_transforms(),
        'keys': ['img', 'mask', 'recon', 'brain_mask'],
        'stage': None,
        'master_seed': 2,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }

@pytest.mark.viz
def test_visualize_transforms():
    """Visualize the reference transforms"""
    transforms = DeterministicCompose(viz_settings['transforms'], 
                                      master_seed=viz_settings['master_seed'])
    visualize_transform_stage(
        pipeline=transforms, 
        sample=sample_data, 
        keys=viz_settings['keys'], 
        stage=viz_settings['stage'], 
        slice_indices=viz_settings['slice_indices'], 
        axis=viz_settings['axis'],
        channel=viz_settings['channel'],
        save_path=viz_settings['save_path'])
