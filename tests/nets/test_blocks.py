"""Test blocks of layers."""

import pytest
import torch

from anyBrainer.networks.blocks import (
    ProjectionHead,
    ClassificationHead,
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
    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    def test_forward(self, bottleneck_output, activation):
        head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation)
        output = head(bottleneck_output)
        assert output.shape == (8, 128)
        assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(8))
    
    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    def test_norm_output(self, bottleneck_output, activation):
        head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128, activation=activation)
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
            head = ProjectionHead(in_dim=768, hidden_dim=2048, proj_dim=128, activation="wrong")
            head(bottleneck_output)


class TestClassificationHead:
    """Test ClassificationHead in terms of shape, and error handling."""
    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize("hidden_dim", [None, 1024])
    def test_forward(self, bottleneck_output, activation, dropout, hidden_dim):
        head = ClassificationHead(in_dim=768, num_classes=7, activation=activation, 
                                dropout=dropout, hidden_dim=hidden_dim)
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