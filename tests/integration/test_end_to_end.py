"""
End-to-end integration tests.
"""
import pytest
import torch
from pathlib import Path

from data.datamodule import SSLDataModule
from utils.mock_data import create_mock_dataset_files

class TestEndToEndIntegration:
    """Test complete workflows from data loading to batch creation."""
    
    def test_mae_pipeline_end_to_end(self, mae_config, temp_data_dir):
        """Test complete MAE pipeline."""
        # Create a more comprehensive dataset
        create_mock_dataset_files(
            temp_data_dir,
            n_subjects=5,
            sessions_per_subject=2,
            modalities=['T1w', 'T2w'],
            volume_shape=(1, 64, 64, 64)
        )
        
        # Create datamodule
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            batch_size=2,
            num_workers=0,
            train_val_test_split=(0.6, 0.2, 0.2)
        )
        
        # Set random state for reproducibility  
        datamodule.set_random_state(42)
        
        # Run complete pipeline
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        # Test training batch
        train_batch = next(iter(train_loader))
        
        # Verify batch structure for MAE
        assert 'img' in train_batch
        assert isinstance(train_batch['img'], torch.Tensor)
        assert train_batch['img'].shape[0] <= 2  # batch_size
        assert len(train_batch['img'].shape) == 5  # (B, C, H, W, D)
        
        # Check if masking transforms were applied
        if mae_config['mae_transforms']['random_mask']['enabled']:
            assert 'mask' in train_batch
            assert isinstance(train_batch['mask'], torch.Tensor)
        
        if mae_config['mae_transforms']['reconstruction_target']['enabled']:
            assert 'recon' in train_batch
        
        if mae_config['mae_transforms']['ignore_mask']['enabled']:
            assert 'ignore' in train_batch
        
        # Test validation batch
        val_batch = next(iter(val_loader))
        assert 'img' in val_batch
        assert isinstance(val_batch['img'], torch.Tensor)
    
    def test_contrastive_pipeline_end_to_end(self, contrastive_config, temp_data_dir):
        """Test complete contrastive learning pipeline."""
        # Create dataset with multiple modalities per session
        create_mock_dataset_files(
            temp_data_dir,
            n_subjects=3,
            sessions_per_subject=2,
            modalities=['T1w', 'T2w', 'FLAIR'],
            volume_shape=(1, 64, 64, 64)
        )
        
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=contrastive_config,
            batch_size=2,
            num_workers=0
        )
        
        datamodule.set_random_state(42)
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        # Test dataloader
        train_loader = datamodule.train_dataloader()
        train_batch = next(iter(train_loader))
        
        # Should have multiple image keys for contrastive learning
        img_keys = [key for key in train_batch.keys() if key.startswith('img_')]
        assert len(img_keys) >= 1
        
        # Each img key should have valid tensor data
        for key in img_keys:
            if key in train_batch and train_batch[key] is not None:
                assert isinstance(train_batch[key], torch.Tensor)
                assert len(train_batch[key].shape) == 5  # (B, C, H, W, D)
    
    def test_configuration_validation(self, temp_data_dir):
        """Test that invalid configurations are caught."""
        invalid_config = {
            'mode': 'invalid_mode',  # Invalid mode
            'patch_size': [64, 64, 64]
        }
        
        with pytest.raises((ValueError, KeyError)):
            datamodule = SSLDataModule(
                data_dir=str(temp_data_dir),
                transform_config=invalid_config,
                num_workers=0
            )
    
    def test_empty_data_directory(self, mae_config):
        """Test behavior with empty data directory."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as empty_dir:
            datamodule = SSLDataModule(
                data_dir=empty_dir,
                transform_config=mae_config,
                num_workers=0
            )
            
            with pytest.raises(FileNotFoundError):
                datamodule.prepare_data()
    
    def test_state_dict_functionality(self, mae_config, temp_data_dir):
        """Test state saving and loading."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            num_workers=0
        )
        
        datamodule.set_random_state(42)
        
        # Get state dict
        state = datamodule.state_dict()
        
        assert 'learning_mode' in state
        assert 'train_val_test_split' in state
        assert 'rng' in state
        
        # Create new datamodule and load state
        new_datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            num_workers=0
        )
        
        new_datamodule.load_state_dict(state)
        
        assert new_datamodule.learning_mode == datamodule.learning_mode
        assert new_datamodule.train_val_test_split == datamodule.train_val_test_split