"""Test networks of blocks."""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.models.networks import Swinv2CL

@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    """Shape (B, C, D, H, W)"""
    return torch.randn(8, 1, 128, 128, 128)

@pytest.fixture(autouse=True)
def mock_swin_vit(monkeypatch):
    """
    Monkey-patch LoadImage so every attempt to read a file
    yields a synthetic 3-D volume instead of touching the disk.
    """
    def _dummy_call(self, *args, **kwargs):
        # Create data with the shape the pipeline expects
        gen = torch.Generator().manual_seed(42)
        x = [torch.rand((8, 768, 4, 4, 4), dtype=torch.float32, generator=gen)]
        return x

    monkeypatch.setattr(SwinViT, "forward", _dummy_call, raising=True)


class TestSwinv2CL:
    def test_forward_all_heads(self, input_tensor):
        model = Swinv2CL(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            feature_size=48,
            proj_dim=128,
            proj_hidden_dim=2048,
            proj_hidden_act="gelu",
            aux_mlp_head=True,
            aux_mlp_num_classes=7,
            aux_mlp_hidden_dim=128,
            aux_mlp_hidden_act="relu",
            aux_mlp_dropout=0.0,
        )
        proj, aux = model(input_tensor)
        assert proj.shape == (8, 128)
        assert aux.shape == (8, 7)
    
    def test_forward_no_aux_mlp(self, input_tensor):
        model = Swinv2CL(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            feature_size=48,
            aux_mlp_head=False,
        )
        proj, aux = model(input_tensor)
        assert proj.shape == (8, 128)
        assert aux is None
