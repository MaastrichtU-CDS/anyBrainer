"""
Test data fixtures and utilities.
"""
import yaml
from pathlib import Path

def create_test_config_file(config_dict, config_path):
    """Create a YAML config file for testing."""
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def load_test_config(config_name: str):
    """Load a test configuration by name."""
    test_configs = {
        'minimal_mae': {
            'mode': 'masked_autoencoder',
            'patch_size': [32, 32, 32],
            'load_transforms': {'load': {'enabled': True}}
        },
        'full_mae': {
            'mode': 'masked_autoencoder', 
            'patch_size': [64, 64, 64],
            'load_transforms': {
                'load': {'enabled': True},
                'pad': {'enabled': True}
            },
            'spatial_transforms': {
                'flip': {'enabled': True, 'prob': 0.5},
                'crop': {'enabled': True}
            },
            'intensity_transforms': {
                'gaussian_noise': {'enabled': True, 'prob': 0.3}
            },
            'mae_transforms': {
                'ignore_mask': {'enabled': True},
                'reconstruction_target': {'enabled': True},
                'random_mask': {'enabled': True}
            }
        },
        'minimal_contrastive': {
            'mode': 'contrastive',
            'patch_size': [32, 32, 32],
            'max_modalities': 5,
            'load_transforms': {'load': {'enabled': True}}
        }
    }
    
    return test_configs.get(config_name, {})