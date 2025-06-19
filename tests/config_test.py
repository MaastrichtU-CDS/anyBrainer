"""
Pytest configuration and shared fixtures for the testing suite.
"""
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
import yaml

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock .npy files for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create mock data files with proper naming convention
    mock_files = [
        "sub_001_ses_001_T1w.npy",
        "sub_001_ses_001_T2w.npy", 
        "sub_001_ses_001_FLAIR.npy",
        "sub_001_ses_002_T1w.npy",
        "sub_002_ses_001_T1w.npy",
        "sub_002_ses_001_T2w.npy",
        "sub_003_ses_001_T1w.npy",
    ]
    
    # Create mock 3D brain data (64x64x64 for speed)
    for filename in mock_files:
        mock_data = np.random.randn(1, 64, 64, 64).astype(np.float32)
        # Add some realistic brain-like structure
        mock_data = np.abs(mock_data) * 1000
        # Add background (zeros) around edges
        mock_data[:, :5, :, :] = 0
        mock_data[:, -5:, :, :] = 0
        mock_data[:, :, :5, :] = 0
        mock_data[:, :, -5:, :] = 0
        mock_data[:, :, :, :5] = 0
        mock_data[:, :, :, -5:] = 0
        
        np.save(temp_path / filename, mock_data)
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mae_config():
    """Basic MAE configuration for testing."""
    return {
        'mode': 'masked_autoencoder',
        'patch_size': [64, 64, 64],
        'load_transforms': {
            'load': {'enabled': True},
            'pad': {'enabled': True}
        },
        'spatial_transforms': {
            'flip': {'enabled': True, 'prob': 0.3},
            'crop': {'enabled': True}
        },
        'intensity_transforms': {
            'gaussian_noise': {'enabled': True, 'prob': 0.2}
        },
        'mae_transforms': {
            'ignore_mask': {'enabled': True},
            'reconstruction_target': {'enabled': True},
            'random_mask': {'enabled': True, 'ratio': 0.6, 'patch_size': 4}
        }
    }

@pytest.fixture  
def contrastive_config():
    """Basic contrastive configuration for testing."""
    return {
        'mode': 'contrastive',
        'patch_size': [64, 64, 64],
        'max_modalities': 10,
        'load_transforms': {
            'load': {'enabled': True},
            'pad': {'enabled': True}
        },
        'spatial_transforms': {
            'flip': {'enabled': True, 'prob': 0.3},
            'crop': {'enabled': True}
        },
        'intensity_transforms': {
            'gaussian_noise': {'enabled': True, 'prob': 0.2}
        }
    }

@pytest.fixture
def minimal_config():
    """Minimal configuration for basic testing."""
    return {
        'mode': 'masked_autoencoder',
        'patch_size': [32, 32, 32],
        'load_transforms': {
            'load': {'enabled': True}
        }
    }