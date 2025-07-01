"""Tests datamodule methods"""

import pytest

from anyBrainer.data.explorer import GenericNiftiDataExplorer
from anyBrainer.data import (
    MAEDataModule
)

@pytest.fixture(autouse=True)
def mock_load_image(monkeypatch):
    """
    Monkey-patch LoadImage so every attempt to read a file
    yields a synthetic 3-D volume instead of touching the disk.
    """
    def _dummy_call(self, as_list, exts, *args, **kwargs):
        return [
            '/Users/project/dataset/sub_1/ses_1/t1.npy',
            '/Users/project/dataset/sub_1/ses_1/t1_2.npy',
            '/Users/project/dataset/sub_1/ses_1/t2.npy',
            '/Users/project/dataset/sub_1/ses_1/dwi.npy',
            '/Users/project/dataset/sub_1/ses_2/t1.npy',
            '/Users/project/dataset/sub_12/ses_1/flair.npy',
            '/Users/project/dataset/sub_123/ses_1/dwi.npy',
            '/Users/project/dataset/sub_123/ses_1/dwi_2.npy',
            '/Users/project/dataset/sub_123/ses_2/dwi.npy',
            '/Users/project/dataset/sub_123/ses_3/dwi.npy',
        ]

    monkeypatch.setattr(
        GenericNiftiDataExplorer, "get_all_image_files", _dummy_call, raising=True
    )

mae_data_settings = {
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
        return MAEDataModule(**mae_data_settings)
    
    def test_data_splits(self, data_module):
        data_module.setup(stage="fit")
        data_module.setup(stage="test")
        assert len(data_module.train_data) == 7
        assert len(data_module.val_data) == 2
        assert len(data_module.test_data) == 1
    
    def test_data_list(self, data_module):
        data_module.setup(stage="fit")
        for i in data_module.train_data:
            assert i.keys() == ['file_name', 'brain_mask', 'sub_id', 
                                'ses_id', 'modality']

