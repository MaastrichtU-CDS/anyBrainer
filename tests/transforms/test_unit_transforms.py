"""Unit tests for masking transforms."""

from copy import deepcopy
from typing import Sequence
import math

import pytest
import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor

from anyBrainer.core.transforms.unit_transforms import (
    CreateRandomMaskd,
    SaveReconstructionTargetd,
    CreateEmptyMaskd,
    GetKeyQueryd,
    SlidingWindowPatch,
    RandImgKeyd,
    ClipNonzeroPercentilesd,
    UnscalePredsIfNeeded,
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


def _compute_num_patches(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    overlap: Sequence[float] | Sequence[int],
) -> int:
    """Compute the number of patches in an image; assumes relative overlap."""
    assert len(image_size) == len(patch_size) == len(overlap), "Length mismatch."

    scan_intervals = []
    for ps, ov in zip(patch_size, overlap):
        stride = max(1, int(ps * (1.0 - ov)))
        scan_intervals.append(stride)

    n_patches_per_dim = [
        math.ceil((dim - ps) / stride) + 1
        for dim, ps, stride in zip(image_size, patch_size, scan_intervals)
    ]
    total_patches = math.prod(n_patches_per_dim)
    return total_patches


class TestSlidingWindowPatch:
    @pytest.mark.parametrize("tensor_shape", [
        (1, 160, 190, 160),
        (4, 1, 160, 190, 160),
        (2, 3, 1, 130, 130, 130),
        (2, 1, 1, 100, 100, 100),
    ])
    def test_different_inputs_same_overlap(self, tensor_shape):
        """
        Test that the output shape (different inputs, same overlap) is correct.
        Indirectly tests that the padding is correct.
        """
        op = SlidingWindowPatch(
            patch_size=128,
            overlap=0.5,
            spatial_dims=3,
        )
        out = op(torch.randn(tensor_shape))
        expected_n_patches = _compute_num_patches(
            image_size=tensor_shape[-3:],
            patch_size=(128, 128, 128),
            overlap=(0.5, 0.5, 0.5),
        )
        assert out.ndim == 1 + len(tensor_shape)
        assert out.shape[-5:] == (expected_n_patches, 1, 128, 128, 128)

    @pytest.mark.parametrize("overlap", [
        (0.25, 0.25, 0.25),
        (0.5, 0.5, 0.5),
        (0.75, 0.75, 0.75),
    ])
    def test_same_input_different_overlap(self, overlap):
        """
        Test that the output shape (same input, different overlap) is correct.
        """
        op = SlidingWindowPatch(
            patch_size=128,
            overlap=overlap,
            spatial_dims=3,
        )
        out = op(torch.randn(1, 160, 190, 160))
        n_patches = _compute_num_patches(
            image_size=(160, 190, 160),
            patch_size=(128, 128, 128),
            overlap=overlap,
        )
        assert out.ndim == 5
        assert out.shape == (n_patches, 1, 128, 128, 128)

    @pytest.mark.parametrize("tensor_shape", [
        (1, 160, 190, 160),
        (4, 1, 160, 190, 160),
        (2, 3, 1, 130, 130, 130),
        (2, 1, 1, 100, 100, 100),
    ])
    def test_different_input_same_n_patches(self, tensor_shape):
        """
        Test that the output shape (different input, same n_patches) is correct.
        """
        op = SlidingWindowPatch(
            patch_size=128,
            overlap=None,
            n_patches=(2, 3, 4),
            spatial_dims=3,
        )
        out = op(torch.randn(tensor_shape))
        assert out.ndim == len(tensor_shape) + 1
        assert out.shape[-5:] == (24, 1, 128, 128, 128)

    @pytest.mark.parametrize("n_patches", [
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
    ])
    def test_same_input_different_n_patches(self, n_patches):
        """
        Test that the output shape (same input, different n_patches) is correct.
        """
        op = SlidingWindowPatch(
            patch_size=128,
            overlap=None,
            n_patches=n_patches,
            spatial_dims=3,
        )
        expected_n_patches = math.prod(n_patches)
        out = op(torch.randn(1, 160, 190, 160))
        assert out.ndim == 5
        assert out.shape == (expected_n_patches, 1, 128, 128, 128)
    
    @pytest.mark.parametrize("tensor_shape", [
        (120, 120, 120),
        (1, 120, 120),
        (1, 120),
    ])
    def test_error_input_shapes(self, tensor_shape):
        """
        Test that the error is raised when the input shape is invalid.
        """
        op = SlidingWindowPatch(
            patch_size=128,
            spatial_dims=3,
        )
        with pytest.raises(ValueError):
            op(torch.randn(tensor_shape))
    
    def test_error_no_patch_or_overlap(self):
        """
        Test that the error is raised when no patch or overlap is provided.
        """
        with pytest.raises(ValueError):
            SlidingWindowPatch(
                patch_size=128,
                spatial_dims=3,
                overlap=None,
            )


class TestRandImgKeyd:
    def test_output_keys(self, contrastive_sample_data):
        out = RandImgKeyd(keys=["img_0", "img_1"])(contrastive_sample_data)
        assert out.get("img_0") is not None
        assert out.get("img_1") is not None
        assert out.get("img") is not None
    
    def test_error_no_keys_match(self, contrastive_sample_data):
        with pytest.raises(KeyError):
            RandImgKeyd(keys=["img_999", "img_1000"])(contrastive_sample_data)
    
    def test_error_no_keys_match_allow_missing(self, contrastive_sample_data):
        out = RandImgKeyd(keys=["img_40", "img_41"], 
                          allow_missing_keys=True)(contrastive_sample_data)
        assert out.get("img_0") is not None
        assert out.get("img_1") is not None
        assert out.get("img") is None
    
    def test_no_error_replace_key(self, contrastive_sample_data):
        out = RandImgKeyd(keys=["img_0", "img_1"], new_key="img_0")(contrastive_sample_data)
        assert out.get("img_0") is not None
        assert out.get("img_1") is not None
        assert out.get("img") is None


class TestClipNonzeroPercentilesd:
    def test_output_keys(self, mae_sample_data):
        out = ClipNonzeroPercentilesd(keys=["img"])(mae_sample_data)
        # All original keys should remain
        assert set(out.keys()) == {"img", "brain_mask", "sub_id", "ses_id", "mod"}

    def test_shapes(self, mae_sample_data):
        out = ClipNonzeroPercentilesd(keys=["img"])(mae_sample_data)
        assert out["img"].shape == mae_sample_data["img"].shape

    def test_type_and_device_preserved(self, mae_sample_data):
        out = ClipNonzeroPercentilesd(keys=["img"])(mae_sample_data)
        # stays a torch tensor (MetaTensor if that’s what you passed)
        assert isinstance(out["img"], torch.Tensor)
        assert out["img"].device == mae_sample_data["img"].device
        # if MetaTensor, metadata should still be there
        if isinstance(mae_sample_data["img"], MetaTensor):
            assert out["img"].meta == mae_sample_data["img"].meta # type: ignore[attr-defined]


class TestUnscalePredsIfNeeded:
    @pytest.mark.parametrize("pred", [
        torch.tensor([-25]),
        torch.tensor([-15]),
        torch.tensor([-5]),
        torch.tensor([5]),
        torch.tensor([15]),
        torch.tensor([25]),
    ])
    def test_offset(self, pred):
        out = UnscalePredsIfNeeded(center=65)(pred)
        assert out == pred + 65
    
    @pytest.mark.parametrize("pred", [
        torch.tensor([-1.0]),
        torch.tensor([-0.5]),
        torch.tensor([0]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    ])
    def test_scale(self, pred):
        out = UnscalePredsIfNeeded(scale_range=(20, 90))(pred)
        assert out == pred * 35
    
    @pytest.mark.parametrize("pred", [
        torch.tensor([-1.0]),
        torch.tensor([-0.5]),
        torch.tensor([0]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    ])
    def test_scale_and_offset(self, pred):
        out = UnscalePredsIfNeeded(scale_range=(20, 90), center=65)(pred)
        assert 30 <= out <= 100
        assert out == pred * 35 + 65