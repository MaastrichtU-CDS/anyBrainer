"""
Visualize transforms.
"""

from anyBrainer.transforms import get_mae_train_transforms
from anyBrainer.transforms import DeterministicCompose
from visualize.utils import visualize_transform_stage


sample_data = {
    "img": 'visualize/examples/t1.npy', 
    "brain_mask": 'visualize/examples/mask.npy',
}

viz_settings = {
        'transforms': get_mae_train_transforms(),
        'keys': ['img', 'mask', 'recon', 'brain_mask'],
        'stage': None,
        'master_seed': 12345,
        'slice_indices': [30, 50, 70],
        'axis': 2,
        'channel': 0,
        'save_path': None,
    }

def visualize_transforms(sample_data, viz_settings):
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

if __name__ == "__main__":
    visualize_transforms(sample_data, viz_settings)
