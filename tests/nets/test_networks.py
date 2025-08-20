"""Test networks of blocks."""

import pytest
import torch
# pyright: reportPrivateImportUsage=false
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

from anyBrainer.core.networks import (
    Swinv2CL, 
    Swinv2Classifier,
    Swinv2ClassifierMidFusion,
)


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
    @pytest.fixture
    def model_w_fusion(self):
        return Swinv2Classifier(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            feature_size=48,
            mlp_num_classes=2,
            mlp_num_hidden_layers=2,
            mlp_hidden_dim=[128, 64],
            mlp_dropout=0.3,
            mlp_activations="LeakyReLU",
            mlp_activation_kwargs={"negative_slope": 0.1},
            late_fusion=True,
            n_late_fusion=4,
        )

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

    @pytest.mark.slow
    @pytest.mark.parametrize("mlp_num_classes", [2, 4, 8, 100])
    def test_forward_w_late_fusion(self, mlp_num_classes):
        """Test model output shape with late fusion."""
        model_w_fusion = Swinv2Classifier(
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
            late_fusion=True,
            n_late_fusion=4,
        )
        output = model_w_fusion(torch.randn(8, 4, 8, 1, 128, 128, 128))
        assert output.shape == (8, mlp_num_classes)

    def test_late_fusion_missing_n_patches(self, model_w_fusion):
        """Test that the model raises an error if the input shape is wrong."""
        with pytest.raises(ValueError):
            model_w_fusion(torch.randn(8, 4, 1, 128, 128, 128))

    def test_late_fusion_wrong_num_modalities(self, model_w_fusion):
        """Test that the model raises an error if the number of modalities is wrong."""
        with pytest.raises(ValueError):
            model_w_fusion(torch.randn(8, 2, 8, 1, 128, 128, 128))
    
    def test_late_fusion_wrong_in_channels(self, model_w_fusion):
        """Test that the model raises an error if the input shape is wrong."""
        with pytest.raises(ValueError):
            model_w_fusion(torch.randn(8, 4, 8, 2, 128, 128, 128))

    def test_late_fusion_wrong_spatial_dims(self, model_w_fusion):
        """Test that the model raises an error if the spatial dimensions are wrong."""
        with pytest.raises(ValueError):
            model_w_fusion(torch.randn(8, 4, 8, 1, 128, 128))


class TestSwinv2ClassifierMidFusion:
    @pytest.mark.parametrize("aggregator", ["noisy_or", "lse", "topk"])
    def test_forward_different_aggregators(self, aggregator):
        """Test that the model can forward pass."""
        model = Swinv2ClassifierMidFusion(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            n_classes=2,
            n_fusion=4,
            aggregator=aggregator,
        )
        output = model(torch.randn(8, 4, 1, 128, 128, 128))
        assert output.shape == (8, 2)
    
    @pytest.mark.parametrize("n_classes", [1, 2, 4, 8])
    def test_forward_different_n_classes(self, n_classes):
        """Test that the model can forward pass."""
        model = Swinv2ClassifierMidFusion(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            n_classes=n_classes,
            n_fusion=4,
        )
        output = model(torch.randn(8, 4, 1, 128, 128, 128))
        assert output.shape == (8, n_classes)
    
    def test_wrong_n_fusion(self):
        """Test that the model raises an error if the number of modalities is wrong."""
        model = Swinv2ClassifierMidFusion(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            n_fusion=4,
        )
        with pytest.raises(ValueError):
            model(torch.randn(8, 3, 1, 128, 128, 128))
    
    def test_wrong_in_channels(self):
        """Test that the model raises an error if the number of input channels is wrong."""
        model = Swinv2ClassifierMidFusion(
            in_channels=3,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            n_fusion=4,
        )
        with pytest.raises(ValueError):
            model(torch.randn(8, 4, 2, 128, 128, 128))
    
    def test_wrong_spatial_dims(self):
        """Test that the model raises an error if the spatial dimensions are wrong."""
        model = Swinv2ClassifierMidFusion(
            in_channels=1,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            patch_size=2,
            use_v2=True,
            spatial_dims=3,
            n_fusion=4,
        )
        with pytest.raises(ValueError):
            model(torch.randn(8, 4, 1, 128, 128))