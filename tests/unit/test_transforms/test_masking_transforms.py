"""
Unit tests for masking transforms.
"""
import pytest
import numpy as np

from transforms.masking_transforms import (
    CreateRandomMaskd,
    CreateIgnoreMaskd,
    SaveReconstructionTargetd
)

class TestCreateRandomMaskd:
    """Test CreateRandomMaskd functionality."""
    
    def test_mask_creation(self):
        """Test basic mask creation."""
        transform = CreateRandomMaskd(
            keys=['img'],
            mask_key='mask',
            mask_ratio=0.5,
            mask_patch_size=4
        )
        
        # Create test data
        test_data = {
            'img': np.random.randn(1, 32, 32, 32).astype(np.float32)
        }
        
        result = transform(test_data)
        
        assert 'mask' in result
        assert result['mask'].shape == (1, 32, 32, 32)
        assert result['mask'].dtype == np.float32
        
        # Check mask ratio is approximately correct
        mask_ratio = np.mean(result['mask'])
        assert 0.3 < mask_ratio < 0.7  # Allow some tolerance
    
    def test_different_mask_ratios(self):
        """Test different mask ratios."""
        for ratio in [0.1, 0.5, 0.9]:
            transform = CreateRandomMaskd(
                keys=['img'],
                mask_ratio=ratio,
                mask_patch_size=4
            )
            
            test_data = {
                'img': np.random.randn(1, 32, 32, 32).astype(np.float32)
            }
            
            result = transform(test_data)
            actual_ratio = np.mean(result['mask'])
            
            # Allow 20% tolerance
            assert abs(actual_ratio - ratio) < 0.2
    
    def test_3d_vs_4d_input(self):
        """Test with both 3D and 4D inputs."""
        transform = CreateRandomMaskd(keys=['img'], mask_patch_size=4)
        
        # Test 3D input
        data_3d = {'img': np.random.randn(32, 32, 32).astype(np.float32)}
        result_3d = transform(data_3d)
        assert result_3d['mask'].shape == (32, 32, 32)
        
        # Test 4D input
        data_4d = {'img': np.random.randn(1, 32, 32, 32).astype(np.float32)}
        result_4d = transform(data_4d)
        assert result_4d['mask'].shape == (1, 32, 32, 32)

class TestCreateIgnoreMaskd:
    """Test CreateIgnoreMaskd functionality."""
    
    def test_background_detection(self):
        """Test background voxel detection."""
        transform = CreateIgnoreMaskd(
            keys=['img'],
            ignore_key='ignore',
            background_threshold=1e-6
        )
        
        # Create data with known background regions
        img = np.ones((1, 32, 32, 32), dtype=np.float32) * 100
        img[:, :5, :, :] = 0  # Background region
        img[:, -5:, :, :] = 1e-8  # Near-zero region
        
        test_data = {'img': img}
        result = transform(test_data)
        
        assert 'ignore' in result
        assert result['ignore'].dtype == np.float32
        
        # Check that background regions are marked as ignore
        assert np.all(result['ignore'][:, :5, :, :] == 1)
        assert np.all(result['ignore'][:, -5:, :, :] == 1)
        assert np.all(result['ignore'][:, 10:20, 10:20, 10:20] == 0)

class TestSaveReconstructionTargetd:
    """Test SaveReconstructionTargetd functionality."""
    
    def test_target_saving(self):
        """Test reconstruction target saving."""
        transform = SaveReconstructionTargetd(
            keys=['img'],
            recon_key='recon'
        )
        
        original_img = np.random.randn(1, 32, 32, 32).astype(np.float32)
        test_data = {'img': original_img.copy()}
        
        result = transform(test_data)
        
        assert 'recon' in result
        np.testing.assert_array_equal(result['recon'], original_img)
        
        # Modify original to ensure copy was made
        result['img'][0, 0, 0, 0] = 999
        assert result['recon'][0, 0, 0, 0] != 999