"""Test engine utilities."""

import pytest

import torch

from anyBrainer.engines.utils import (
    get_sub_ses_tensors,
    pack_ids,
    sync_dist_safe,
)
from anyBrainer.utils.data import (
    modality_to_onehot,
)
from anyBrainer.utils.models import (
    get_total_grad_norm,
)

@pytest.fixture(scope="module")
def input_batch():
    """Input batch for the model."""
    return {
        "query": torch.randn(8, 1, 128, 128, 128),
        "key": torch.randn(8, 1, 128, 128, 128),
        "mod": ["t1", "t2", "flair", "dwi", "adc", "swi", "other", "t1"],
        "sub_id": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-01"],
        "ses_id": ["ses-01", "ses-01", "ses-01", "ses-02", "ses-01", "ses-01", "ses-01", "ses-01"],
    }

def test_modality_to_onehot(input_batch):
    """Test that the modality to one-hot encoding is correct."""
    one_hot = modality_to_onehot(input_batch, "mod", torch.device("cpu"))
    assert one_hot.shape == (8, 7)
    assert one_hot.dtype == torch.float32
    assert torch.all(one_hot == torch.tensor([
        [1, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 1], 
        [1, 0, 0, 0, 0, 0, 0],
    ]))

@pytest.mark.parametrize("sub, ses", [
    (["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-01"], 
    ["ses-01", "ses-01", "ses-01", "ses-02", "ses-01", "ses-01", "ses-01", "ses-01"]),
    ([1, 2, 3, 4, 5, 6, 7, 1], 
    [1, 1, 1, 2, 1, 1, 1, 1]),
    (torch.tensor([1, 2, 3, 4, 5, 6, 7, 1]), 
    torch.tensor([1, 1, 1, 2, 1, 1, 1, 1])),
    (["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-01"], 
    [1, 1, 1, 2, 1, 1, 1, 1]),
    (["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-01"], 
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

@pytest.mark.skip(reason="Not implemented")
def test_sync_dist_safe():
    """Test that the sync_dist_safe function is correct."""