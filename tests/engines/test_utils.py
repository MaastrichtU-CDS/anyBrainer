"""Test engine utilities."""

from pathlib import Path

import pytest

import torch

from anyBrainer.core.engines.utils import (
    get_sub_ses_tensors,
    pack_ids,
)
from anyBrainer.core.utils import (
    modality_to_onehot,
    get_total_grad_norm,
    load_param_group_from_ckpt,
)
from anyBrainer.core.networks import Swinv2Classifier

@pytest.fixture(autouse=True)
def mock_torch_load(monkeypatch):
    """Mock the torch.load function."""
    def mock_load(*args, **kwargs):
        return {
            "state_dict": {
                "encoder.patch_embed.proj.weight": torch.randn(48, 1, 128, 128, 128),
                "encoder.patch_embed.proj.bias": torch.randn(48),
                "encoder.layers1.0.blocks.0.attn.qkv.weight": torch.randn(144, 48),
                "encoder.layers1.0.blocks.0.attn.qkv.bias": torch.randn(144),
                "encoder.layers1.0.blocks.0.attn.proj.weight": torch.randn(48, 48),
                "encoder.layers1.0.blocks.0.attn.proj.bias": torch.randn(48),

                "encoder.weight": torch.randn(128),
                "encoder.bias": torch.randn(128),

                "classifier.weight": torch.randn(2, 128),
                "classifier.bias": torch.randn(2),
            }
        }
    monkeypatch.setattr("torch.load", mock_load)

@pytest.fixture(scope="module")
def input_batch():
    """Input batch for the model."""
    return {
        "query": torch.randn(8, 1, 128, 128, 128),
        "key": torch.randn(8, 1, 128, 128, 128),
        "mod": ["t1", "t2", "flair", "dwi", "other", "other", "other", "t1"],
        "sub_id": ["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "sub_06", "sub_07", "sub_01"],
        "ses_id": ["ses_01", "ses_01", "ses_01", "ses_02", "ses_01", "ses_01", "ses_01", "ses_01"],
    }

@pytest.fixture(scope="module")
def classification_model_w_encoder():
    """Classification model with encoder."""
    model = Swinv2Classifier(
        patch_size=128,
        mlp_num_classes=2,
        mlp_num_hidden_layers=1,
        mlp_hidden_dim=64,
        mlp_dropout=0.1,
        mlp_activations="GELU",
    )
    return model

def test_modality_to_onehot(input_batch):
    """Test that the modality to one-hot encoding is correct."""
    one_hot = modality_to_onehot(input_batch, "mod", torch.device("cpu"))
    assert one_hot.shape == (8, 5)
    assert one_hot.dtype == torch.float32
    assert torch.all(one_hot == torch.tensor([
        [1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [0, 0, 1, 0, 0], 
        [0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1], 
        [1, 0, 0, 0, 0],
    ]))

@pytest.mark.parametrize("sub, ses", [
    (["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "sub_06", "sub_07", "sub_01"], 
    ["ses_01", "ses_01", "ses_01", "ses_02", "ses_01", "ses_01", "ses_01", "ses_01"]),
    ([1, 2, 3, 4, 5, 6, 7, 1], 
    [1, 1, 1, 2, 1, 1, 1, 1]),
    (torch.tensor([1, 2, 3, 4, 5, 6, 7, 1]), 
    torch.tensor([1, 1, 1, 2, 1, 1, 1, 1])),
    (["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "sub_06", "sub_07", "sub_01"], 
    [1, 1, 1, 2, 1, 1, 1, 1]),
    (["sub_01", "sub_02", "sub_03", "sub_04", "sub_05", "sub_06", "sub_07", "sub_01"], 
    torch.tensor([1, 1, 1, 2, 1, 1, 1, 1])),
])
def test_sub_ses_ids(input_batch, sub, ses):
    """Test that the subject and session ID tensors are returned correctly."""
    input_batch["sub_id"] = sub
    input_batch["ses_id"] = ses
    sub_id, ses_id = get_sub_ses_tensors(input_batch, torch.device("cpu"))
    assert sub_id.shape == (8,)
    assert ses_id.shape == (8,)
    assert sub_id.dtype == torch.int64
    assert ses_id.dtype == torch.int64
    assert torch.all(sub_id == torch.tensor([1, 2, 3, 4, 5, 6, 7, 1]))
    assert torch.all(ses_id == torch.tensor([1, 1, 1, 2, 1, 1, 1, 1]))

def test_pack_ids(input_batch):
    """Test that the pack_ids function is correct."""
    sub_id, ses_id = get_sub_ses_tensors(input_batch, torch.device("cpu"))
    packed = pack_ids(sub_id, ses_id)
    assert packed.shape == (8,)
    assert packed.dtype == torch.int64
    assert torch.all(packed == torch.tensor(
        [1e6 + 1, 2e6 + 1, 3e6 + 1, 4e6 + 2, 5e6 + 1, 6e6 + 1, 7e6 + 1, 1e6 + 1]
    ))

def test_get_total_norm(model_with_grads):
    """Compare with reference grad norm using a dummy optimized model"""
    norm = get_total_grad_norm(model_with_grads)

    ref_norm = torch.norm(
        torch.stack([
            p.grad.detach().norm(2)
            for p in model_with_grads.parameters()
            if p.grad is not None
        ]), p=2,
    )

    assert torch.equal(norm, ref_norm)

def test_load_encoder_from_checkpoint(classification_model_w_encoder):
    """Test that the encoder is loaded correctly."""
    _, stats = load_param_group_from_ckpt(
        model_instance=classification_model_w_encoder,
        checkpoint_path=Path("test.pt"),
        select_prefixes="encoder",
    )
    print(stats)
    assert len(stats["loaded_keys"]) == 8
    assert len(stats["unexpected_keys"]) == 2