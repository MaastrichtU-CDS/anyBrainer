"""Tests for get_segmentation_transforms options."""

import pytest
from monai.transforms.croppad.dictionary import CenterSpatialCropd, SpatialPadd
from monai.transforms.spatial.dictionary import RandAffined

from anyBrainer.core.transforms.transformlists import get_segmentation_transforms
from anyBrainer.core.transforms.utils import (
    resolve_seg_image_pad_modes,
    scale_spatial_size,
)


class TestSegImagePadModes:
    @pytest.mark.parametrize(
        ("pad_img", "affine", "spatial"),
        [
            ("zeros", "zeros", "constant"),
            ("border", "border", "edge"),
        ],
    )
    def test_resolve(self, pad_img, affine, spatial):
        assert resolve_seg_image_pad_modes(pad_img) == (affine, spatial)

    def test_invalid_pad_img(self):
        with pytest.raises(ValueError, match="pad_img must be one of"):
            resolve_seg_image_pad_modes("reflect")  # type: ignore[arg-type]


class TestScaleSpatialSize:
    def test_int_input_size(self):
        assert scale_spatial_size(128, 1.2) == (154, 154, 154)

    def test_sequence_input_size(self):
        assert scale_spatial_size((100, 110, 120), 1.1) == (110, 121, 132)

    def test_invalid_factor(self):
        with pytest.raises(
            ValueError, match="pre_spatial_scale_factor must be positive"
        ):
            scale_spatial_size(128, 0.0)


class TestGetSegmentationTransforms:
    def _find(self, steps, cls):
        return [t for t in steps if isinstance(t, cls)]

    def test_defaults_match_legacy_padding(self):
        steps = get_segmentation_transforms(keys=["ch1", "ch2"], seg_key="label")
        affine = self._find(steps, RandAffined)[0]
        assert list(affine.padding_mode) == ["border", "border", "constant"]

    def test_pad_img_zeros(self):
        steps = get_segmentation_transforms(
            keys=["ch1"], seg_key="label", pad_img="zeros"
        )
        affine = self._find(steps, RandAffined)[0]
        assert list(affine.padding_mode) == ["zeros", "constant"]

    def test_pre_spatial_scale_factor_adds_pad_and_crop(self):
        steps = get_segmentation_transforms(
            keys=["ch1"],
            seg_key="label",
            input_size=128,
            pre_spatial_scale_factor=1.2,
            val_mode=False,
        )
        spatial_pads = self._find(steps, SpatialPadd)
        crops = self._find(steps, CenterSpatialCropd)
        assert len(spatial_pads) >= 2
        assert spatial_pads[0].spatial_size == (154, 154, 154)
        assert len(crops) == 1
        assert crops[0].roi_size == (128, 128, 128)

    def test_pre_spatial_scale_factor_skipped_in_val_mode(self):
        steps = get_segmentation_transforms(
            keys=["ch1"],
            seg_key="label",
            pre_spatial_scale_factor=1.2,
            val_mode=True,
        )
        assert self._find(steps, RandAffined) == []
        assert len(self._find(steps, CenterSpatialCropd)) == 1
