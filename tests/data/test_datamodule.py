"""Tests datamodule methods"""

from pathlib import Path

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    LoadImage,
    Compose,
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.transforms import Randomizable

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
    def _dummy_call(self, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(42)
        img = torch.rand((1, 120, 120, 120), dtype=torch.float32, generator=gen)
        return img

    monkeypatch.setattr(LoadImage, "__call__", _dummy_call, raising=True)

data_settings = {
    'data_dir': '/Users/project/dataset',
    'masks_dir': '/Users/project/masks',
    'batch_size': 2, 
    'num_workers': 4, 
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
            assert set(i.keys()) == {'img', 'sub_id', 'ses_id', 'mod'}
    
    @pytest.mark.slow
    def test_train_loader_rng_locks(self, data_module):
        # Datamodule
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)
        next_out = next(train_loader_iter)
        
        # Ensure outputs don't match between iterations
        for i, j in zip(out['img'], next_out['img']): # type: ignore
            assert not torch.equal(i, j)
    
    @pytest.mark.slow
    def test_train_loader_w_reference(self, data_module, ref_mae_train_transforms):
        # Datamodule
        set_determinism(seed=data_settings['seed'])
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())

        # Reference transforms
        set_determinism(seed=12345)
        ref_dataset = Dataset(
            data=data_module.train_data, 
            transform=Compose(ref_mae_train_transforms).set_random_state(seed=12345)
        )
        ref_loader_iter = iter(DataLoader(ref_dataset, batch_size=data_settings['batch_size'], 
                                num_workers=data_settings['num_workers'], shuffle=True))
        for _ in range(2):
            out = next(train_loader_iter)
            ref_out = next(ref_loader_iter)
        
        # Ensure outputs match for each sample
        for i, j in zip(out['img'], ref_out['img']): # type: ignore
            assert torch.equal(i, j)


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
    
    @pytest.mark.slow
    def test_train_loader_pair(self, data_module):
        set_determinism(seed=data_settings['seed'])
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)
        
        # Ensure keys and queries are different
        for i, j in zip(out['key'], out['query']): # type: ignore
            assert not torch.equal(i, j)
    
    @pytest.mark.slow
    def test_train_loader_rng_locks(self, data_module):
        # Datamodule
        set_determinism(seed=data_settings['seed'])
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())
        out = next(train_loader_iter)
        next_out = next(train_loader_iter)
        
        # Ensure outputs don't match between iterations
        for i, j in zip(out['key'], next_out['key']): # type: ignore
            assert not torch.equal(i, j)

    @pytest.mark.slow
    def test_train_loader_w_reference(self, data_module, ref_contrastive_train_transforms):
        # Datamodule
        set_determinism(seed=data_settings['seed'])
        data_module.setup(stage="fit")
        train_loader_iter = iter(data_module.train_dataloader())

        # Reference transforms
        set_determinism(seed=12345)
        ref_dataset = Dataset(
            data=data_module.train_data, 
            transform=Compose(ref_contrastive_train_transforms).set_random_state(seed=12345)
        )
        ref_loader_iter = iter(DataLoader(ref_dataset, batch_size=data_settings['batch_size'], 
                                num_workers=data_settings['num_workers'], shuffle=True))
        for _ in range(2):
            out = next(train_loader_iter)
            ref_out = next(ref_loader_iter)
        
        # Ensure outputs match for each sample
        for i, j in zip(out['key'], ref_out['key']): # type: ignore
            assert torch.equal(i, j)
        
        for i, j in zip(out['query'], ref_out['query']): # type: ignore
            assert torch.equal(i, j)