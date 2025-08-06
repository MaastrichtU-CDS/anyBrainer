"""Test inferers."""

import math
from typing import Sequence
import pytest
import torch
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.core.networks import Swinv2Classifier
from anyBrainer.core.inferers import SlidingWindowClassificationInferer

@pytest.fixture(scope="module")
def input_batch_classification():
    """Input batch for the model."""
    return {
        "img_0": torch.randn(8, 1, 160, 190, 160),
        "label": torch.randint(0, 2, (8, 2)),
    }


@pytest.fixture(autouse=True)
def mock_swin_vit(monkeypatch):
    """
    Monkey-patch MONAI's SwinTransformer so every forward pass of the model
    yields a synthetic tensor that matches the bottleneck dimensions, given 
    a feature_size of 48 and 4 stages.
    """
    def _dummy_call(self, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(42)
        x = [torch.rand((8, 768, 4, 4, 4), dtype=torch.float32, generator=gen)]
        return x

    monkeypatch.setattr(SwinViT, "forward", _dummy_call, raising=True)

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

class TestSlidingWindowClassificationInferer:
    @pytest.fixture
    def model(self):
        """Model fixture."""
        return Swinv2Classifier(
            patch_size=2,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            feature_size=48,
            use_v2=True,
            mlp_num_classes=2,
            mlp_num_hidden_layers=1,
            mlp_hidden_dim=128,
            mlp_dropout=0.3,
            mlp_activations="GELU",
            mlp_activation_kwargs={"dropout": 0.3},
        )
    
    @pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
    def test_sliding_window_shape(self, model, input_batch_classification, overlap):
        """Test that the model is properly initialized."""
        inferer = SlidingWindowClassificationInferer(
            model=model,
            patch_size=(128, 128, 128),
            overlap=overlap,
            aggregation_mode="none",
        )
        n_patches = _compute_num_patches(
            image_size=(160, 190, 160),
            patch_size=(128, 128, 128),
            overlap=(overlap, overlap, overlap),
        )
        preds = inferer(input_batch_classification["img_0"], model)
        assert preds.shape == (8, n_patches, 2)
    
    @pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
    def test_sliding_window_weighted(self, model, input_batch_classification, overlap):
        """Test that the model is properly initialized."""
        inferer = SlidingWindowClassificationInferer(
            model=model,
            patch_size=(128, 128, 128),
            overlap=overlap,
            aggregation_mode="weighted",
        )
        preds = inferer(input_batch_classification["img_0"], model)
        assert preds.shape == (8, 2)
    
    def test_sliding_window_mean(self, model, input_batch_classification):
        """Test that the model is properly initialized."""
        inferer = SlidingWindowClassificationInferer(
            model=model,
            patch_size=(128, 128, 128),
            overlap=0.5,
            aggregation_mode="mean",
        )
        preds = inferer(input_batch_classification["img_0"], model)
        assert preds.shape == (8, 2)
    
    def test_sliding_window_majority(self, model, input_batch_classification):
        """Test that the model is properly initialized."""
        inferer = SlidingWindowClassificationInferer(
            model=model,
            patch_size=(128, 128, 128),
            overlap=0.5,
            aggregation_mode="majority",
        )
        preds = inferer(input_batch_classification["img_0"], model)
        assert preds.shape == (8, 2)