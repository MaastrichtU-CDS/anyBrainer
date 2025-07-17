"""Tests for Pytorch's Lightning Module."""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.models.model import Swinv2CLModel

@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    """Shape (B, C, D, H, W)"""
    return torch.randn(8, 1, 128, 128, 128)

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

swinv2cl_model_kwargs = {
    "in_channels": 1,
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
    "window_size": 7,
    "patch_size": 2,
    "use_v2": True,
    "feature_size": 48,
    "proj_dim": 128,
    "proj_hidden_dim": 2048,
    "proj_hidden_act": "gelu",
    "aux_mlp_head": True,
    "aux_mlp_num_classes": 7,
}

class TestSwinv2CLModel:
    def test_forward(self, input_tensor):
        """Test that the model is properly initialized."""
        model = Swinv2CLModel(model_kwargs=swinv2cl_model_kwargs)
        proj, aux = model(input_tensor)
        assert proj.shape == (8, 128)
        assert aux.shape == (8, 7)
