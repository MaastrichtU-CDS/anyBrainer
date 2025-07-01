"""Tests datamodule methods"""

from pathlib import Path

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    LoadImage,
)

from anyBrainer.data.explorer import GenericNiftiDataExplorer
from anyBrainer.data import (
    MAEDataModule,
    ContrastiveDataModule,
)

@pytest.fixture(autouse=True)
def mock_data_explorer(monkeypatch):
    """
    Monkey-patch LoadImage so every attempt to read a file
    yields a synthetic 3-D volume instead of touching the disk.
    """
    def _dummy_call(self, *args, **kwargs): 
        # 10 subjects, 11 sessions, 15 images
        return [
            Path('/Users/project/dataset/sub_1/ses_1/t1.npy'),
            Path('/Users/project/dataset/sub_1/ses_1/t1_2.npy'),
            Path('/Users/project/dataset/sub_1/ses_1/t2.npy'),
            Path('/Users/project/dataset/sub_1/ses_1/dwi.npy'),
            Path('/Users/project/dataset/sub_1/ses_2/t1.npy'),
            Path('/Users/project/dataset/sub_12/ses_1/flair.npy'),
            Path('/Users/project/dataset/sub_123/ses_1/dwi.npy'),
            Path('/Users/project/dataset/sub_123/ses_1/dwi_2.npy'),
            Path('/Users/project/dataset/sub_1234/ses_2/dwi.npy'),
            Path('/Users/project/dataset/sub_1235/ses_1/t1.npy'),
            Path('/Users/project/dataset/sub_2235/ses_11/dwi.npy'),
            Path('/Users/project/dataset/sub_3235/ses_1/flair.npy'),
            Path('/Users/project/dataset/sub_4235/ses_1/t2.npy'),
            Path('/Users/project/dataset/sub_5235/ses_2/dwi.npy'),
            Path('/Users/project/dataset/sub_6235/ses_3/dwi.npy'),
        ]

    monkeypatch.setattr(
        GenericNiftiDataExplorer, "get_all_image_files", _dummy_call, raising=True
    )

@pytest.fixture(autouse=True)
def mock_load_image(monkeypatch):
    """
    Monkey-patch LoadImage so every attempt to read a file
    yields a synthetic 3-D volume instead of touching the disk.
    """
    def _dummy_call(self, filename, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(hash(filename) & 0xFFFF_FFFF)
        img = torch.rand((1, 120, 120, 120), dtype=torch.float32, generator=gen)
        # LoadImage normally returns (np.ndarray, meta_dict)
        return img

    monkeypatch.setattr(LoadImage, "__call__", _dummy_call, raising=True)

data_settings = {
    'data_dir': '/Users/project/dataset',
    'masks_dir': '/Users/project/masks',
    'batch_size': 8, 
    'num_workers': 32, 
    'train_val_test_split': (0.7, 0.2, 0.1),
    'seed': 12345

}

class TestMAEDataModule: 
    @pytest.fixture
    def data_module(self):
        return MAEDataModule(**data_settings)
    
    def test_data_splits(self, data_module):
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        assert len(data_module.train_data) == 11 # specific to seed; sub_1 (5 scans), ...
        assert len(data_module.val_data) == 2 # specific to seed; ...
        assert len(data_module.test_data) == 2 # specific to seed; sub_123 (2 scans), ...
    
    def test_data_list(self, data_module):
        data_module.setup(stage="fit")
        for i in data_module.train_data:
            assert set(i.keys()) == {'file_name', 'sub_id', 'ses_id', 'modality'}


class TestContrastiveDataModule: 
    @pytest.fixture
    def data_module(self):
        return ContrastiveDataModule(**data_settings)

    def test_data_splits(self, data_module):
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        assert len(data_module.train_data) == 8 # specific to seed; sub_1 (2 sessions), ...
        assert len(data_module.val_data) == 2 # specific to seed; ...
        assert len(data_module.test_data) == 1 # specific to seed; ...

