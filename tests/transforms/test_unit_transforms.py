"""Unit tests for masking transforms."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d

from anyBrainer.transforms.unit_transforms import (
    CreateRandomMaskd,
    SaveReconstructionTargetd,
    EmptyMaskd,
)

@pytest.mark.usefixtures("sample_data")
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
    
    def test_output_keys(self, op, sample_data):
        out = op(sample_data)
        assert out.keys() == {"img", "img_1", "mask", "brain_mask", "sub_id", 
                              "ses_id", "mod", "count"}

    def test_shapes(self, op, sample_data):
        out = op(sample_data)
        assert out["img"].shape == (1, 120, 120, 120)
        assert out["mask"].shape == (1, 120, 120, 120)

    def test_mask_ratio(self, op, sample_data, mask_ratio):
        """NOTE: it fails with mask_patch_size > img_size/2 due to expected rounding errors"""
        out = op(sample_data)
        mask_ratio_actual = out["mask"].float().mean().item()
        assert (1- mask_ratio) - 0.1 < mask_ratio_actual < (1- mask_ratio) + 0.1, \
            f"Expected {mask_ratio} +/- 0.1 mask ratio, got {mask_ratio_actual}"
    
    def test_patch_uniformity(self, op, sample_data, mask_patch_size):
        """
        Check that every non-overlapping patch of size mask_patch_size^3
        is entirely masked or entirely un-masked.
        """
        out = op(sample_data)
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


@pytest.mark.usefixtures("sample_data")
class TestSaveReconstructionTargetd:
    @pytest.fixture
    def op(self):
        return SaveReconstructionTargetd(recon_key="recon")
    
    def test_output_keys(self, op, sample_data):
        out = op(sample_data)
        assert out.keys() == {"img", "img_1", "recon", "brain_mask", "sub_id", 
                              "ses_id", "mod", "count"}
    
    def test_shapes(self, op, sample_data):
        out = op(sample_data)
        assert out["img"].shape == (1, 120, 120, 120)
        assert out["recon"].shape == (1, 120, 120, 120)


@pytest.mark.usefixtures("sample_data")
class TestEmptyMaskd:
    @pytest.fixture
    def get_data(self, sample_data):
        data = deepcopy(sample_data)
        data.pop("brain_mask")
        return data
    
    def test_skip_if_exists(self, sample_data):
        out = EmptyMaskd(mask_key="brain_mask")(sample_data)
        ref = torch.zeros_like(sample_data["img"])
        assert (out["brain_mask"] != ref).any()
    
    def test_mask_key(self, get_data):
        out = EmptyMaskd(mask_key="brain_mask")(get_data)
        assert "brain_mask" in out
    
    def test_empty_mask(self, get_data):
        out = EmptyMaskd(mask_key="brain_mask")(get_data)
        ref = torch.zeros_like(get_data["img"])
        assert (out["brain_mask"] == ref).all()
    
    def test_empty_mask_shape(self, get_data):
        out = EmptyMaskd(mask_key="brain_mask")(get_data)
        assert out["brain_mask"].shape == get_data["img"].shape
    