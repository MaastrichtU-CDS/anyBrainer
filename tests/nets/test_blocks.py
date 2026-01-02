"""Test blocks of layers."""

import pytest
import torch

from anyBrainer.core.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
    FusionHead,
    FPNLightDecoder3D,
)


@pytest.fixture(scope="module")
def bottleneck_output() -> torch.Tensor:
    """Shape (B, C, D, H, W)"""
    return torch.randn(8, 768, 4, 4, 4)


@pytest.fixture(scope="module")
def bottleneck_output_flat() -> torch.Tensor:
    """Shape (B, C)"""
    return torch.randn(8, 768)


@pytest.fixture(scope="module")
def bottleneck_output_invalid() -> torch.Tensor:
    """Shape (B, C, D, H)"""
    return torch.randn(8, 768, 4, 4)


class TestProjectionHead:
    """Test ProjectionHead in terms of shape, norm, and error handling."""

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "SiLU"])
    def test_forward(self, bottleneck_output, activation):
        head = ProjectionHead(
            in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation
        )
        output = head(bottleneck_output)
        assert output.shape == (8, 128)
        assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(8))

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "SiLU"])
    def test_norm_output(self, bottleneck_output, activation):
        head = ProjectionHead(
            in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation
        )
        output = head(bottleneck_output)
        assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(8))

    def test_forward_flat(self, bottleneck_output_flat):
        head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128)
        output = head(bottleneck_output_flat)
        assert output.shape == (8, 128)

    def test_forward_error(self, bottleneck_output_invalid):
        with pytest.raises(ValueError):
            head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128)
            head(bottleneck_output_invalid)

    def test_invalid_activation(self, bottleneck_output):
        with pytest.raises(ValueError):
            head = ProjectionHead(
                in_dim=768, hidden_dim=2048, proj_dim=128, activation="wrong"
            )
            head(bottleneck_output)


class TestClassificationHead:
    """Test ClassificationHead in terms of shape, and error handling."""

    @pytest.mark.parametrize(
        "num_hidden_layers, hidden_dim, activation, activation_kwargs, dropout",
        [
            (0, 0, "ReLU", None, 0.0),
            (1, 1024, "GELU", None, 0.5),
            (2, [512, 64], "SiLU", {"inplace": False}, 0.5),
        ],
    )
    def test_forward(
        self,
        bottleneck_output,
        activation,
        dropout,
        hidden_dim,
        num_hidden_layers,
        activation_kwargs,
    ):
        head = ClassificationHead(
            in_dim=768,
            num_classes=7,
            activation=activation,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation_kwargs=activation_kwargs,
        )
        output = head(bottleneck_output)
        assert output.shape == (8, 7)

    def test_forward_flat(self, bottleneck_output_flat):
        head = ClassificationHead(in_dim=768, num_classes=7)
        output = head(bottleneck_output_flat)
        assert output.shape == (8, 7)

    def test_forward_error(self, bottleneck_output_invalid):
        with pytest.raises(ValueError):
            head = ClassificationHead(in_dim=768, num_classes=7)
            head(bottleneck_output_invalid)

    @pytest.mark.parametrize(
        "num_hidden_layers, hidden_dim", [(0, [1024]), (2, [512, 64, 16])]
    )
    def test_error_hidden_dim_args(self, num_hidden_layers, hidden_dim):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation="ReLU",
                dropout=0.5,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "num_hidden_layers, dropout", [(1, [0.5]), (2, [0.5, 0.25])]
    )
    def test_error_dropout_args(self, num_hidden_layers, dropout):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation="ReLU",
                dropout=dropout,
                hidden_dim=1024,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "num_hidden_layers, activation", [(1, ["ReLU"]), (2, ["ReLU", "GELU"])]
    )
    def test_error_activation_args(self, num_hidden_layers, activation):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation=activation,
                dropout=0.5,
                hidden_dim=1024,
                num_hidden_layers=num_hidden_layers,
            )

    @pytest.mark.parametrize(
        "activation, activation_kwargs",
        [(["ReLU", "GELU"], [None]), ("SiLU", [{"inplace": False}])],
    )
    def test_error_activation_kwargs_args(self, activation, activation_kwargs):
        with pytest.raises(ValueError):
            ClassificationHead(
                in_dim=768,
                num_classes=7,
                activation=activation,
                dropout=0.5,
                hidden_dim=1024,
                num_hidden_layers=1,
                activation_kwargs=activation_kwargs,
            )


class TestFusionHead:
    """Test FusionHead in terms of shape, and error handling."""

    @pytest.mark.parametrize(
        "input_shape, n_fusion",
        [
            ((4, 4, 8, 768, 4, 4, 4), 4),
            ((4, 8, 768, 4, 4, 4), 1),
            ((4, 768, 4, 4, 4), 1),
        ],
    )
    def test_forward(self, input_shape, n_fusion):
        fusion_head = FusionHead(n_fusion=n_fusion)
        output = fusion_head(torch.randn(input_shape))  # type: ignore
        assert output.shape == (4, 768)

    def test_mod_only(self):
        fusion_head = FusionHead(n_fusion=4, mod_only=True)
        output = fusion_head(torch.randn(4, 4, 768))  # type: ignore
        assert output.shape == (4, 768)

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2)
            fusion_head(torch.randn(768, 4, 4, 4))  # type: ignore

    def test_wrong_shape_mod_only(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2, mod_only=True)
            fusion_head(torch.randn(4, 4, 8, 768, 4, 4, 4))  # type: ignore

    def test_mismatch_n_fusion(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=2)
            fusion_head(torch.randn(4, 3, 8, 768, 4, 4, 4))  # type: ignore

    def test_mismatch_n_fusion_mod_only(self):
        with pytest.raises(ValueError):
            fusion_head = FusionHead(n_fusion=4, mod_only=True)
            fusion_head(torch.randn(4, 3, 768))  # type: ignore


class TestFPNLightDecoder3D:
    """Test FPNLightDecoder3D in terms of shape, and error handling."""

    @pytest.fixture
    def decoder(self):
        return FPNLightDecoder3D(
            in_feats=[48, 96, 192, 384, 768], in_chans=1, out_channels=2
        )

    @pytest.mark.slow
    def test_forward(self, decoder):
        output = decoder(
            torch.randn(4, 1, 64, 64, 64),
            [
                torch.randn(4, 48, 32, 32, 32),
                torch.randn(4, 96, 16, 16, 16),
                torch.randn(4, 192, 8, 8, 8),
                torch.randn(4, 384, 4, 4, 4),
                torch.randn(4, 768, 2, 2, 2),
            ],
        )
        assert output.shape == (4, 2, 64, 64, 64)

    def test_wrong_shape_feats(self, decoder):
        with pytest.raises(ValueError):
            decoder(
                torch.randn(4, 1, 64, 64, 64),
                [
                    torch.randn(4, 48, 32, 32, 32),
                    torch.randn(4, 96, 16, 16, 16),
                    torch.randn(4, 192, 8, 8, 8),
                ],
            )
