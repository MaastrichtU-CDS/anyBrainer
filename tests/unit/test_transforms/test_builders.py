"""
Unit tests for transform builders.
"""
import pytest
import numpy as np
# pyright: reportPrivateImportUsage=false
from monai.transforms import Compose

from transforms.builders import (
    LoadTransformBuilder,
    SpatialTransformBuilder,
    IntensityTransformBuilder,
    MaskingTransformBuilder
)

class TestLoadTransformBuilder:
    """Test LoadTransformBuilder functionality."""
    
    def test_basic_build(self, minimal_config):
        """Test basic transform building."""
        builder = LoadTransformBuilder(minimal_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)
        assert len(transform.transforms) >= 1
    
    def test_with_multiple_keys(self, minimal_config):
        """Test building with multiple image keys."""
        builder = LoadTransformBuilder(minimal_config)
        transform = builder.build(['img_0', 'img_1', 'img_2'])
        
        assert isinstance(transform, Compose)
    
    def test_allow_missing_keys(self, minimal_config):
        """Test allow_missing_keys parameter."""
        builder = LoadTransformBuilder(minimal_config)
        transform = builder.build(['img'], allow_missing_keys=True)
        
        # Should not raise error
        assert isinstance(transform, Compose)

class TestSpatialTransformBuilder:
    """Test SpatialTransformBuilder functionality."""
    
    def test_basic_build(self, mae_config):
        """Test basic spatial transform building."""
        builder = SpatialTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)
    
    def test_disabled_transforms(self, mae_config):
        """Test with some transforms disabled."""
        # Disable flip transform
        mae_config['spatial_transforms']['flip']['enabled'] = False
        
        builder = SpatialTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)
        # Should have fewer transforms than if all were enabled

class TestIntensityTransformBuilder:
    """Test IntensityTransformBuilder functionality."""
    
    def test_basic_build(self, mae_config):
        """Test basic intensity transform building."""
        builder = IntensityTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)
    
    def test_with_custom_params(self, mae_config):
        """Test with custom parameters."""
        mae_config['intensity_transforms']['gaussian_noise']['params'] = {
            'std': 0.05,
            'prob': 0.5
        }
        
        builder = IntensityTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)

class TestMaskingTransformBuilder:
    """Test MaskingTransformBuilder functionality."""
    
    def test_basic_build(self, mae_config):
        """Test basic masking transform building."""
        builder = MaskingTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        assert isinstance(transform, Compose)
    
    def test_mask_generation(self, mae_config, temp_data_dir):
        """Test that masking transforms actually generate masks."""
        from utils.mock_data import create_mock_brain_volume
        
        builder = MaskingTransformBuilder(mae_config)
        transform = builder.build(['img'])
        
        # Create test data
        mock_data = {
            'img': create_mock_brain_volume((1, 32, 32, 32))
        }
        
        # Apply transforms
        result = transform(mock_data)
        
        # Should have created additional keys
        assert 'img' in result
        # Check if masking keys were added (depends on enabled transforms)
        if mae_config['mae_transforms']['random_mask']['enabled']:
            assert 'mask' in result