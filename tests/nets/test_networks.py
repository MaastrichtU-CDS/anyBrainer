"""Test networks of blocks."""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.core.networks import Swinv2CL, Swinv2Classifier


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


class TestSwinv2CL:
    def test_forward_all_heads(self, input_tensor):
        """Test that the heads are working as expected."""
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
            proj_hidden_act="GELU",
            aux_mlp_head=True,
            aux_mlp_num_classes=7,
            aux_mlp_hidden_dim=128,
            aux_mlp_hidden_act="ReLU",
            aux_mlp_dropout=0.0,
        )
        proj, aux = model(input_tensor)
        assert proj.shape == (8, 128)
        assert aux.shape == (8, 7)
    
    def test_forward_no_aux_mlp(self, input_tensor):
        """Test that the model can skip the auxiliary MLP."""
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


class TestSwinv2Classifier:
    @pytest.mark.parametrize("mlp_num_classes", [2, 4, 8, 100])
    def test_forward(self, input_tensor, mlp_num_classes):
        """Test that the model can forward pass."""
        model = Swinv2Classifier(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            feature_size=48,
            mlp_num_classes=mlp_num_classes,
            mlp_num_hidden_layers=2,
            mlp_hidden_dim=[128, 64],
            mlp_dropout=0.3,
            mlp_activations="LeakyReLU",
            mlp_activation_kwargs={"negative_slope": 0.1},
        )
        output = model(input_tensor)
        assert output.shape == (8, mlp_num_classes)