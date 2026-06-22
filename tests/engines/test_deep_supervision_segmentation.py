"""Tests for DeepSupervisionSegmentationModel."""

import pytest
import torch
import torch.nn as nn

from anyBrainer.core.engines.models import DeepSupervisionSegmentationModel
from anyBrainer.factories.unit import UnitFactory


class StackedOutputNet(nn.Module):
    """Stub network returning stacked deep-supervision outputs."""

    def __init__(self, num_heads: int = 3, out_channels: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, modality=()) -> torch.Tensor:
        b = x.size(0)
        spatial = x.shape[2:]
        return torch.randn(
            b,
            self.num_heads,
            self.out_channels,
            *spatial,
            device=x.device,
            dtype=x.dtype,
        )


@pytest.fixture
def ds_seg_model(monkeypatch):
    stub = StackedOutputNet(num_heads=3, out_channels=1)

    def _fake_get_model(cls, model_kwargs):
        return stub

    monkeypatch.setattr(
        UnitFactory,
        "get_model_instance_from_kwargs",
        classmethod(_fake_get_model),
    )

    model = DeepSupervisionSegmentationModel(
        model_kwargs={"name": "StackedOutputNet"},
        loss_fn_kwargs={"name": "MSELoss"},
        optimizer_kwargs={"name": "SGD", "lr": 0.01},
        metrics=[],
    )
    monkeypatch.setattr(model, "log_step", lambda *args, **kwargs: None)
    return model


class TestDeepSupervisionSegmentationModel:
    def test_resolve_ds_weights_zero_coarsest(self, ds_seg_model):
        weights = ds_seg_model._resolve_ds_weights(3)
        assert len(weights) == 3
        assert weights[-1] == pytest.approx(0.0)
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] > weights[1]

    def test_resolve_ds_weights_custom(self, ds_seg_model):
        ds_seg_model.ds_weights = [0.5, 0.3, 0.2]
        ds_seg_model.zero_coarsest_weight = False
        weights = ds_seg_model._resolve_ds_weights(3)
        assert weights == pytest.approx([0.5, 0.3, 0.2])

    def test_split_outputs_stacked(self, ds_seg_model):
        target = torch.zeros(2, 1, 4, 4, 4)
        stacked = torch.zeros(2, 3, 1, 4, 4, 4)
        heads = ds_seg_model._split_outputs(stacked, target)
        assert len(heads) == 3
        assert all(h.shape == target.shape for h in heads)

    def test_split_outputs_single(self, ds_seg_model):
        target = torch.zeros(2, 1, 4, 4, 4)
        single = torch.zeros(2, 1, 4, 4, 4)
        heads = ds_seg_model._split_outputs(single, target)
        assert len(heads) == 1
        assert heads[0] is single

    def test_compute_loss_weighted(self, ds_seg_model):
        target = torch.zeros(2, 1, 4, 4, 4)
        heads = [
            torch.ones_like(target),
            torch.ones_like(target) * 2,
            torch.zeros_like(target),
        ]
        stacked = torch.stack(heads, dim=1)
        weights = ds_seg_model._resolve_ds_weights(3)

        expected = sum(
            w * ds_seg_model.loss_fn(h, target) for w, h in zip(weights, heads)
        )
        actual = ds_seg_model.compute_loss(stacked, target)
        assert actual == pytest.approx(expected)

    def test_training_step_uses_main_head_for_metrics(self, ds_seg_model, monkeypatch):
        batch = {
            "img": torch.randn(2, 1, 4, 4, 4),
            "label": torch.zeros(2, 1, 4, 4, 4),
        }
        captured: dict[str, torch.Size] = {}

        def _fake_compute_metrics(out, target):
            captured["out_shape"] = out.shape
            return {"loss": 0.0}

        monkeypatch.setattr(ds_seg_model, "compute_metrics", _fake_compute_metrics)
        ds_seg_model.training_step(batch, 0)
        assert captured["out_shape"] == batch["label"].shape
