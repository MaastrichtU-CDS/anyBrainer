"""Tests for transform managers"""

from typing import cast
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
    get_classification_train_transforms,
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


@pytest.mark.slow
class TestClassificationTrainTransforms:
    def test_output_keys_default(self, contrastive_sample_data):
        """Tests default output keys"""
        op = Compose(get_classification_train_transforms())
        out = cast(dict, op(contrastive_sample_data))
        assert set(out.keys()) == {
            'img_0', 'img_1', 'img_2', 'sub_id', 'ses_id', 'mod_0', 'mod_1', 'mod_2', 'count'
        }

    def test_output_shapes_default(self, contrastive_sample_data):
        """Tests patch_size arg"""
        op = Compose(get_classification_train_transforms(patch_size=120))
        out = cast(dict, op(contrastive_sample_data))
        assert out['img_0'].shape == (1, 120, 120, 120)
        assert out['img_1'].shape == (1, 120, 120, 120)
        assert out['img_2'].shape == (1, 120, 120, 120)

    def test_output_types_default(self, contrastive_sample_data):
        """Tests default output types"""
        op = Compose(get_classification_train_transforms())
        out = cast(dict, op(contrastive_sample_data))
        assert isinstance(out['img_0'], MetaTensor)
        assert isinstance(out['img_1'], MetaTensor)
        assert isinstance(out['img_2'], MetaTensor)

    def test_output_keys_concat(self, contrastive_sample_data):
        """Tests if concat_img=True replaces img_* with img"""
        op = Compose(get_classification_train_transforms(concat_img=True))
        out = cast(dict, op(contrastive_sample_data))
        assert set(out.keys()) == {
            'img', 'sub_id', 'ses_id', 'mod_0', 'mod_1', 'mod_2', 'count'
        }
    
    def test_output_shapes_concat(self, contrastive_sample_data):
        """Tests if concat_img=True gives 5D tensor (n_patches, n_mod, *patch_size)"""
        op = Compose(get_classification_train_transforms(concat_img=True, patch_size=120))
        out = cast(dict, op(contrastive_sample_data))
        assert out['img'].ndim == 5
        assert out['img'].shape == (1, 3, 120, 120, 120)
    
    def test_output_types_concat(self, contrastive_sample_data):
        """Tests if concat_img=True preserves MetaTensor"""
        op = Compose(get_classification_train_transforms(concat_img=True))
        out = cast(dict, op(contrastive_sample_data))
        assert isinstance(out['img'], MetaTensor)
    
    @pytest.mark.parametrize("patch_size_int,patch_size_tuple", [
        (32, (32, 32, 32)),
        (64, (64, 64, 64)),
        (144, (144, 144, 144)),
    ])
    def test_ensure_tuple_dim(self, contrastive_sample_data, patch_size_int, patch_size_tuple):
        """Tests if ensure_tuple_dim() works"""
        op_int = Compose(get_classification_train_transforms(patch_size=patch_size_int, concat_img=True))
        op_tuple = Compose(get_classification_train_transforms(patch_size=patch_size_tuple, concat_img=True))
        out_int = cast(dict, op_int(contrastive_sample_data))
        out_tuple = cast(dict, op_tuple(contrastive_sample_data))
        assert out_int['img'].shape == out_tuple['img'].shape

    def test_missing_keys_error(self, contrastive_sample_data):
        """Tests allow_missing_keys=False arg"""
        op = Compose(get_classification_train_transforms(allow_missing_keys=False))
        with pytest.raises(RuntimeError):
            op(contrastive_sample_data)