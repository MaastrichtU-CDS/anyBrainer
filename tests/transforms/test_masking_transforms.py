"""Unit tests for masking transforms."""

import pytest
import torch
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

@pytest.mark.parametrize("mask_ratio,mask_patch_size", [
    (0.1, 4),
    (0.3, 8),
    (0.5, 4),
    (0.5, 16),
    (0.7, 8),
    (0.7, 32),
])
def test_create_random_maskd(sample_item, mask_ratio, mask_patch_size):
    """Test the CreateRandomMaskd transform."""
    transform = CreateRandomMaskd(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
    transformed = transform(sample_item)

    # Check shape
    assert transformed["img"].shape == (1, 120, 120, 120)
    assert transformed["mask"].shape == (1, 120, 120, 120)

    # Check mask ratio
    mask_ratio_actual = transformed["mask"].float().mean().item()
    assert (1- mask_ratio) - 0.1 < mask_ratio_actual < (1- mask_ratio) + 0.1, \
        f"Expected {mask_ratio} +/- 0.1 mask ratio, got {mask_ratio_actual}"
