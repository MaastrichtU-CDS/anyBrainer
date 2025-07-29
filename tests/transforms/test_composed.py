"""Tests for transform managers"""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.utils import set_determinism
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImage,
)

from anyBrainer.core.transforms import (
    get_mae_train_transforms,
    get_mae_val_transforms,
    get_contrastive_train_transforms,
    get_contrastive_val_transforms,
)

set_determinism(seed=12345)

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
        # LoadImage normally returns (np.ndarray, meta_dict)
        return img

    monkeypatch.setattr(LoadImage, "__call__", _dummy_call, raising=True)
    

@pytest.mark.slow
class TestMAETrainTransforms:
    def test_output_keys(self, mae_sample_data):
        op = Compose(get_mae_train_transforms())
        out = op(mae_sample_data)
        assert set(out.keys()) == {'img', 'brain_mask', 'mask', 'recon', # type: ignore
                                   'sub_id', 'ses_id', 'mod'} # type: ignore
    
    def test_output_types(self, mae_sample_data):
        op = Compose(get_mae_train_transforms())
        out = op(mae_sample_data)
        assert isinstance(out['img'], MetaTensor) # type: ignore
        assert isinstance(out['brain_mask'], MetaTensor) # type: ignore
        assert isinstance(out['mask'], MetaTensor) # type: ignore
        assert isinstance(out['recon'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['mod'], str) # type: ignore
    
    def test_output_values(self, ref_mae_train_transforms, mae_sample_data):
        t1 = Compose(get_mae_train_transforms())
        t2 = Compose(ref_mae_train_transforms)
        out = t1(mae_sample_data)
        out_expected = t2(mae_sample_data)
        
        assert torch.equal(out['img'], out_expected['img']) # type: ignore
        assert torch.equal(out['brain_mask'], out_expected['brain_mask']) # type: ignore
        assert torch.equal(out['mask'], out_expected['mask']) # type: ignore
        assert torch.equal(out['recon'], out_expected['recon']) # type: ignore

@pytest.mark.slow
class TestMAEValTransforms:
    def test_output_keys(self, mae_sample_data):
        op = Compose(get_mae_val_transforms())
        out = op(mae_sample_data)
        assert set(out.keys()) == {'img', 'brain_mask', 'mask', 'recon', # type: ignore
                                   'sub_id', 'ses_id', 'mod'} # type: ignore
    
    def test_output_types(self, mae_sample_data):
        op = Compose(get_mae_val_transforms())
        out = op(mae_sample_data)
        assert isinstance(out['img'], MetaTensor) # type: ignore
        assert isinstance(out['brain_mask'], MetaTensor) # type: ignore
        assert isinstance(out['mask'], MetaTensor) # type: ignore
        assert isinstance(out['recon'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['mod'], str) # type: ignore
    
    def test_output_values(self, ref_mae_val_transforms, mae_sample_data):
        t1 = Compose(get_mae_val_transforms())
        t2 = Compose(ref_mae_val_transforms)
        out = t1(mae_sample_data)
        out_expected = t2(mae_sample_data)
        
        assert torch.equal(out['img'], out_expected['img']) # type: ignore
        assert torch.equal(out['brain_mask'], out_expected['brain_mask']) # type: ignore
        assert torch.equal(out['mask'], out_expected['mask']) # type: ignore
        assert torch.equal(out['recon'], out_expected['recon']) # type: ignore



@pytest.mark.slow
class TestContrastiveTrainTransforms:
    def test_output_keys(self, contrastive_sample_data):
        op = Compose(get_contrastive_train_transforms())
        out = op(contrastive_sample_data)
        assert set(out.keys()) == {'query', 'key', 'sub_id', 'ses_id', 'mod'} # type: ignore

    def test_output_types(self, contrastive_sample_data):
        op = Compose(get_contrastive_train_transforms())
        out = op(contrastive_sample_data)
        assert isinstance(out['query'], MetaTensor) # type: ignore
        assert isinstance(out['key'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['mod'], str) # type: ignore
    
    def test_output_values(self, ref_contrastive_train_transforms, contrastive_sample_data):
        t1 = Compose(get_contrastive_train_transforms())
        t2 = Compose(ref_contrastive_train_transforms)
        out = t1(contrastive_sample_data)
        out_expected = t2(contrastive_sample_data)
        
        assert torch.equal(out['query'], out_expected['query']) # type: ignore
        assert torch.equal(out['key'], out_expected['key']) # type: ignore
    
@pytest.mark.slow
class TestContrastiveValTransforms:
    def test_output_keys(self, contrastive_sample_data):
        op = Compose(get_contrastive_val_transforms())
        out = op(contrastive_sample_data)
        assert set(out.keys()) == {'query', 'key', 'sub_id', 'ses_id', 'mod'} # type: ignore

    def test_output_types(self, contrastive_sample_data):
        op = Compose(get_contrastive_val_transforms())
        out = op(contrastive_sample_data)
        assert isinstance(out['query'], MetaTensor) # type: ignore
        assert isinstance(out['key'], MetaTensor) # type: ignore
        assert isinstance(out['sub_id'], str) # type: ignore
        assert isinstance(out['ses_id'], str) # type: ignore
        assert isinstance(out['mod'], str) # type: ignore
    
    def test_output_values(self, ref_contrastive_val_transforms, contrastive_sample_data):
        t1 = Compose(get_contrastive_val_transforms())
        t2 = Compose(ref_contrastive_val_transforms)
        out = t1(contrastive_sample_data)
        out_expected = t2(contrastive_sample_data)
        
        assert torch.equal(out['query'], out_expected['query']) # type: ignore
        assert torch.equal(out['key'], out_expected['key']) # type: ignore