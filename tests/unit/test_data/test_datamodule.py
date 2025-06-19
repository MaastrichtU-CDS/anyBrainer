"""
Unit tests for data modules.
"""
import pytest
from pathlib import Path

from data.datamodule import SSLDataModule

class TestSSLDataModule:
    """Test SSLDataModule functionality."""
    
    def test_init_mae_mode(self, mae_config, temp_data_dir):
        """Test initialization in MAE mode."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            batch_size=2,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        assert datamodule.learning_mode == 'masked_autoencoder'
        assert datamodule.patch_size == (64, 64, 64)
        assert datamodule.batch_size == 2
    
    def test_init_contrastive_mode(self, contrastive_config, temp_data_dir):
        """Test initialization in contrastive mode."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=contrastive_config,
            batch_size=4,
            num_workers=0
        )
        
        assert datamodule.learning_mode == 'contrastive'
        assert datamodule.batch_size == 4
    
    def test_prepare_data(self, mae_config, temp_data_dir):
        """Test prepare_data method."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            num_workers=0
        )
        
        # Should not raise any exceptions
        datamodule.prepare_data()
    
    def test_filename_parsing(self, mae_config, temp_data_dir):
        """Test filename parsing functionality."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            num_workers=0
        )
        
        # Test with a known filename
        test_file = temp_data_dir / "sub_001_ses_001_T1w.npy"
        metadata = datamodule._parse_filename(test_file)
        
        assert metadata is not None
        assert metadata['sub_id'] == '001'
        assert metadata['ses_id'] == '001'
        assert metadata['modality'] == 'T1w'
        assert metadata['count'] == 1
    
    def test_setup_mae_mode(self, mae_config, temp_data_dir):
        """Test setup method in MAE mode.""" 
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            num_workers=0
        )
        
        datamodule.set_random_state(42)  # For reproducible tests
        datamodule.setup('fit')
        
        assert datamodule.train_data is not None
        assert datamodule.val_data is not None
        assert len(datamodule.train_data) > 0
        assert len(datamodule.val_data) > 0
    
    def test_setup_contrastive_mode(self, contrastive_config, temp_data_dir):
        """Test setup method in contrastive mode."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=contrastive_config,
            num_workers=0
        )
        
        datamodule.set_random_state(42)
        datamodule.setup('fit')
        
        assert datamodule.train_data is not None
        assert datamodule.val_data is not None
    
    def test_dataloaders(self, mae_config, temp_data_dir):
        """Test dataloader creation."""
        datamodule = SSLDataModule(
            data_dir=str(temp_data_dir),
            transform_config=mae_config,
            batch_size=2,
            num_workers=0
        )
        
        datamodule.set_random_state(42)
        datamodule.prepare_data()
        datamodule.setup('fit')
        
        # Test train dataloader
        train_loader = datamodule.train_dataloader()
        assert train_loader is not None
        
        # Test validation dataloader
        val_loader = datamodule.val_dataloader()
        assert val_loader is not None
        
        # Test that we can iterate through at least one batch
        train_batch = next(iter(train_loader))
        assert 'img' in train_batch
        assert train_batch['img'].shape[0] <= 2  # batch_size
    
    def test_random_state_reproducibility(self, mae_config, temp_data_dir):
        """Test that setting random state gives reproducible results."""
        # Create two identical datamodules
        dm1 = SSLDataModule(str(temp_data_dir), mae_config, num_workers=0)
        dm2 = SSLDataModule(str(temp_data_dir), mae_config, num_workers=0)
        
        # Set same random state
        dm1.set_random_state(42)
        dm2.set_random_state(42)
        
        # Setup both
        dm1.setup('fit')
        dm2.setup('fit')
        
        # Should have same train/val split
        assert len(dm1.train_data) == len(dm2.train_data)
        assert len(dm1.val_data) == len(dm2.val_data)
        
        # Extract subject IDs from both datasets
        train_subjects_1 = set(item['sub_id'] for item in dm1.train_data)
        train_subjects_2 = set(item['sub_id'] for item in dm2.train_data)
        
        assert train_subjects_1 == train_subjects_2