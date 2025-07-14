"""Unit tests for masking transforms."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from anyBrainer.transforms.unit_transforms import (
    CreateRandomMaskd,
    SaveReconstructionTargetd,
    CreateEmptyMaskd,
    GetKeyQueryd,
)

@pytest.mark.parametrize("mask_ratio,mask_patch_size", [
        (0.1, 4),
        (0.3, 8),
        (0.5, 16),
        (0.7, 60),
    ])
class TestCreateRandomMaskd:
    @pytest.fixture
    def op(self, mask_ratio, mask_patch_size):
        return CreateRandomMaskd(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    
    def test_output_keys(self, op, mae_sample_data):
        out = op(mae_sample_data)
        assert out.keys() == {"img", "mask", "brain_mask", "sub_id", 
                              "ses_id", "mod"}

    def test_shapes(self, op, mae_sample_data):
        out = op(mae_sample_data)
        assert out["img"].shape == (1, 120, 120, 120)
        assert out["mask"].shape == (1, 120, 120, 120)

    def test_mask_ratio(self, op, mae_sample_data, mask_ratio):
        """NOTE: it fails with mask_patch_size > img_size/2 due to expected rounding errors"""
        out = op(mae_sample_data)
        mask_ratio_actual = out["mask"].float().mean().item()
        assert (1- mask_ratio) - 0.1 < mask_ratio_actual < (1- mask_ratio) + 0.1, \
            f"Expected {mask_ratio} +/- 0.1 mask ratio, got {mask_ratio_actual}"
    
    def test_patch_uniformity(self, op, mae_sample_data, mask_patch_size):
        """
        Check that every non-overlapping patch of size mask_patch_size^3
        is entirely masked or entirely un-masked.
        """
        out = op(mae_sample_data)
        mask = out["mask"].float()

        mask_5d = mask.unsqueeze(0) # add batch dim for pooling: (N, C, D, H, W)
        pooled = F.avg_pool3d(
            mask_5d,
            kernel_size=mask_patch_size,
            stride=mask_patch_size,
            ceil_mode=False, # ignore incomplete edge patches
        )

        assert torch.all(
            (pooled == 0) | (pooled == 1)
        ), "Found a patch that is only partially masked"


class TestSaveReconstructionTargetd:
    @pytest.fixture
    def op(self):
        return SaveReconstructionTargetd(recon_key="recon")
    
    def test_output_keys(self, op, mae_sample_data):
        out = op(mae_sample_data)
        assert out.keys() == {"img", "recon", "brain_mask", "sub_id", 
                              "ses_id", "mod"}
    
    def test_shapes(self, op, mae_sample_data):
        out = op(mae_sample_data)
        assert out["img"].shape == (1, 120, 120, 120)
        assert out["recon"].shape == (1, 120, 120, 120)


class TestEmptyMaskd:
    @pytest.fixture
    def get_data(self, mae_sample_data):
        data = deepcopy(mae_sample_data)
        data.pop("brain_mask")
        return data
    
    def test_skip_if_exists(self, mae_sample_data):
        out = CreateEmptyMaskd(mask_key="brain_mask")(mae_sample_data)
        ref = torch.zeros_like(mae_sample_data["img"])
        assert (out["brain_mask"] != ref).any()
    
    def test_mask_key(self, get_data):
        out = CreateEmptyMaskd(mask_key="brain_mask")(get_data)
        assert "brain_mask" in out
    
    def test_empty_mask(self, get_data):
        out = CreateEmptyMaskd(mask_key="brain_mask")(get_data)
        ref = torch.zeros_like(get_data["img"])
        assert (out["brain_mask"] == ref).all()
    
    def test_empty_mask_shape(self, get_data):
        out = CreateEmptyMaskd(mask_key="brain_mask")(get_data)
        assert out["brain_mask"].shape == get_data["img"].shape


class TestGetKeyQueryd:
    def test_output_keys_random(self, contrastive_sample_data):
        out = GetKeyQueryd(keys_prefix="img", count_key="count")(contrastive_sample_data)
        assert out.keys() == {"query", "key", "sub_id", "ses_id", "mod"}
    
    def test_output_keys_augmented(self, contrastive_sample_data):
        out = GetKeyQueryd(always_augment_query=True, query_key="img_0", 
                           extra_keys=["mod_0", "sub_id", "ses_id"])(contrastive_sample_data)
        assert out.keys() == {"query", "key", "sub_id", "ses_id", "mod_0"}
    
    def test_output_values_random(self, contrastive_sample_data):
        out = GetKeyQueryd(keys_prefix="img", count_key="count", track=True)(contrastive_sample_data)
        assert (out["query"] == contrastive_sample_data[f"img_{out['track']['query_idx']}"]).all()
        assert (out["key"] == contrastive_sample_data[f"img_{out['track']['key_idx']}"]).all()
    
    def test_output_values_augmented(self, contrastive_sample_data):
        out = GetKeyQueryd(always_augment_query=True, query_key="img_0", 
                           extra_keys=["mod_0", "sub_id", "ses_id"])(contrastive_sample_data)
        assert (out["query"] == contrastive_sample_data["img_0"]).all()
        assert (out["key"] == contrastive_sample_data["img_0"]).all()