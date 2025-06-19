"""Unit tests for masking transforms."""

import pytest
import torch
import torch.nn.functional as F
# pyright: reportPrivateImportUsage=false
from monai.data import create_test_image_3d

from anyBrainer.transforms.masking_transforms import (
    CreateRandomMaskd,
    SaveReconstructionTargetd,
)

@pytest.fixture(scope="module")
def sample_item():
    img, seg = create_test_image_3d(120, 120, 120, channel_dim=0)
    return {
        "img": torch.tensor(img), 
        "brain_mask": torch.tensor(seg).long(),
        "sub_id": "1",
        "ses_id": "1",
        "modality": "t1",
        "count": 2,
    }

@pytest.mark.usefixtures("sample_item")
@pytest.mark.parametrize("mask_ratio,mask_patch_size", [
        (0.1, 4),
        (0.3, 8),
        (0.5, 16),
        (0.7, 32),
    ])
class TestCreateRandomMaskd:
    @pytest.fixture
    def op(self, mask_ratio, mask_patch_size):
        return CreateRandomMaskd(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    
    def check_outputs(self, op, sample_item):
        out = op(sample_item)
        assert out.keys() == {"img", "mask", "brain_mask", "sub_id", 
                              "ses_id", "modality", "count"}

    def test_shapes(self, op, sample_item):
        out = op(sample_item)
        assert out["img"].shape == (1, 120, 120, 120)
        assert out["mask"].shape == (1, 120, 120, 120)

    def test_mask_ratio(self, op, sample_item, mask_ratio):
        out = op(sample_item)
        mask_ratio_actual = out["mask"].float().mean().item()
        assert (1- mask_ratio) - 0.1 < mask_ratio_actual < (1- mask_ratio) + 0.1, \
            f"Expected {mask_ratio} +/- 0.1 mask ratio, got {mask_ratio_actual}"
    
    def test_patch_uniformity(self, op, sample_item, mask_patch_size):
        """
        Check that every non-overlapping patch of size
        `mask_patch_size`³ is entirely masked or entirely un-masked.
        """
        out = op(sample_item)
        mask = out["mask"].float()

        # add batch dim for pooling: (N, C, D, H, W)
        mask_5d = mask.unsqueeze(0)
        pooled = F.avg_pool3d(
            mask_5d,
            kernel_size=mask_patch_size,
            stride=mask_patch_size,
            ceil_mode=False,   # ignore incomplete edge patches
        )

        assert torch.all(
            (pooled == 0) | (pooled == 1)
        ), "Found a patch that is only partially masked"