"""
Tests for transform managers
TODO: Implement MAE validation transforms
"""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d
from monai.utils import set_determinism
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImage,
)

from anyBrainer.transforms import (
    DeterministicCompose,
)

from anyBrainer.transforms.managers import (
    MAETransformManager,
    ContrastiveTransformManager,
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

@pytest.fixture(scope="module")
def sample_config():
    return {
        'patch_size': (128, 128, 128),
        'load_transforms': {
            'do_all': True,
        },
        'spatial_transforms': {
            'do_all': True,
        },
        'intensity_transforms': {
            'do_all': True,
        },
        'mae_transforms': {
            'do_all': True,
        },
        'patch_size_2': (128, 128, 128),
        'load_transforms_2': {
            'do_all': True,
        },
        'spatial_transforms_2': {
            'do_all': True,
        },
        'intensity_transforms_2': {
            'do_all': True,
        },
    }


set_determinism(seed=12345)


@pytest.mark.slow
@pytest.mark.skip(reason="Not implemented")
class TestMAETransformManager:
    @pytest.fixture
    def mae_transforms(self, sample_config):
        return MAETransformManager(sample_config).get_train_transforms()
    
    def test_compose(self, mae_transforms):
        transforms = mae_transforms
        assert isinstance(transforms, Compose)
    
    def test_output_keys(self, mae_transforms, mae_sample_data):
        transforms = mae_transforms
        out = transforms(mae_sample_data)
        assert set(out.keys()) == {'img', 'brain_mask', 'mask', 'recon',
                                   'sub_id', 'ses_id', 'mod'}

    def test_output_types(self, mae_transforms, mae_sample_data):
        transforms = mae_transforms
        out = transforms(mae_sample_data)
        assert isinstance(out['img'], MetaTensor)
        assert isinstance(out['brain_mask'], MetaTensor)
        assert isinstance(out['mask'], MetaTensor)
        assert isinstance(out['sub_id'], str)
        assert isinstance(out['ses_id'], str)
        assert isinstance(out['mod'], str)
    
    def test_output_values(self, ref_mae_train_transforms, mae_transforms, mae_sample_data):
        transforms = DeterministicCompose(mae_transforms, master_seed=12345)
        ref_transforms = DeterministicCompose(ref_mae_train_transforms, master_seed=12345)
        out = transforms(mae_sample_data)
        out_expected = ref_transforms(mae_sample_data)
        
        assert torch.equal(out['img'], out_expected['img']) # type: ignore
        assert torch.equal(out['brain_mask'], out_expected['brain_mask']) # type: ignore
        assert torch.equal(out['mask'], out_expected['mask']) # type: ignore
        assert torch.equal(out['recon'], out_expected['recon']) # type: ignore


@pytest.mark.slow
@pytest.mark.skip(reason="Not implemented")
class TestMAETransformManagerVal:
    @pytest.fixture
    def mae_val_transforms(self, sample_config):
        return MAETransformManager(sample_config).get_val_transforms()

    def test_compose(self, mae_val_transforms):
        assert isinstance(mae_val_transforms, Compose)

    def test_output_keys(self, mae_val_transforms, mae_sample_data):
        out = mae_val_transforms(mae_sample_data)
        assert set(out.keys()) == {"img", "brain_mask", "mask", "recon",
                                   "sub_id", "ses_id", "mod"}

    def test_output_types(self, mae_val_transforms, mae_sample_data):
        out = mae_val_transforms(mae_sample_data)
        assert isinstance(out["img"], MetaTensor)   
        assert isinstance(out["brain_mask"], MetaTensor)
        assert isinstance(out["mask"], MetaTensor)

    def test_output_values(self, ref_mae_val_transforms, sample_config, mae_sample_data):
        transforms = DeterministicCompose(MAETransformManager(sample_config).get_val_transforms(), master_seed=12345)
        ref_transforms = DeterministicCompose(ref_mae_val_transforms, master_seed=12345)
        out = transforms(mae_sample_data)
        out_expected = ref_transforms(mae_sample_data)
        assert torch.equal(out["img"], out_expected["img"])  # type: ignore
        assert torch.equal(out["brain_mask"], out_expected["brain_mask"])  # type: ignore
        assert torch.equal(out["mask"], out_expected["mask"])  # type: ignore
        assert torch.equal(out["recon"], out_expected["recon"])  # type: ignore