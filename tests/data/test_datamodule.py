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
        ]

    monkeypatch.setattr(
        GenericNiftiDataExplorer, "get_all_image_files", _dummy_call, raising=True
    )

mae_data_settings = {
    'data_dir': '/Users/project/dataset',
    'masks_dir': '/Users/project/masks',
    'batch_size': 8, 
    'num_workers': 32, 
    'train_val_test_split': (0.7, 0.15, 0.15),
    'seed': 12345

}

class TestMAEDataModule: 
    @pytest.fixture
    def data_module():
        return MAEDataModule(**mae_data_settings)
