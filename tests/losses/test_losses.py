"""Test loss functions."""

import pytest
import torch
import torch.nn.functional as F

from anyBrainer.losses import (
    InfoNCELoss,
)

@pytest.fixture(scope="module")
def query_tensor() -> torch.Tensor:
    """Shape (B, D)"""
    return torch.randn(8, 128)

@pytest.fixture(scope="module")
def key_tensor() -> torch.Tensor:
    """Shape (B, D)"""
    return torch.randn(8, 128)

@pytest.fixture(scope="module")
def queue_tensor() -> torch.Tensor:
    """Shape (K, D)"""
    return torch.randn(80, 128)

class TestInfoNCELoss:
    def test_forward_default(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with default parameters."""
        loss = InfoNCELoss()
        loss_value, _ = loss(query_tensor, key_tensor, queue_tensor)
        assert loss_value.shape == ()
        assert loss_value.item() > 0
    
    def test_forward_cl_stats(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with default parameters."""
        loss = InfoNCELoss()
        _, cl_stats = loss(query_tensor, key_tensor, queue_tensor)
        assert cl_stats["pos_mean"].shape == ()
        assert cl_stats["neg_mean"].shape == ()
        assert cl_stats["contrastive_acc"].shape == ()
        assert cl_stats["neg_entropy"].shape == ()
    
    def test_forward_top_k_negatives(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with top_k_negatives parameter."""
        loss = InfoNCELoss(top_k_negatives=10)
        loss_value, _ = loss(query_tensor, key_tensor, queue_tensor)
        assert loss_value.shape == ()
        assert loss_value.item() > 0
    
    def test_forward_top_k_negatives_exceeds_queue_size(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with top_k_negatives parameter exceeding queue size."""
        loss = InfoNCELoss(top_k_negatives=100)
        with pytest.raises(ValueError):
            loss(query_tensor, key_tensor, queue_tensor)
        
    def test_forward_top_k_negatives_zero(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with top_k_negatives parameter set to zero."""
        loss = InfoNCELoss(top_k_negatives=0)
        with pytest.raises(ValueError):
            loss(query_tensor, key_tensor, queue_tensor)
    
    def test_forward_postprocess_fn(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with postprocess_fn parameter."""
        def postprocess_fn(l_pos, l_neg):
            return l_pos, l_neg
        loss = InfoNCELoss(postprocess_fn=postprocess_fn)
        loss_value, _ = loss(query_tensor, key_tensor, queue_tensor)
        assert loss_value.shape == ()
        assert loss_value.item() > 0
    
    def test_forward_cross_entropy_args(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with cross_entropy_args parameter."""
        loss = InfoNCELoss(cross_entropy_args={"reduction": "none"})
        loss_value, _ = loss(query_tensor, key_tensor, queue_tensor)
        assert loss_value.shape == (8,)
        assert loss_value.sum().item() > 0
    
    def test_forward_w_reference(self, query_tensor, key_tensor, queue_tensor):
        """Compare loss with a reference implementation."""
        pos_logits = (query_tensor * key_tensor).sum(dim=1, keepdim=True)
        neg_logits = query_tensor @ queue_tensor.T
        ref_loss = F.cross_entropy(torch.cat([pos_logits, neg_logits], dim=1) / 0.07, 
                                   torch.zeros(8, dtype=torch.long))
        loss = InfoNCELoss()
        loss_value, _ = loss(query_tensor, key_tensor, queue_tensor)
        assert torch.equal(loss_value, ref_loss)    
    
    def test_forward_w_reference_cl_stats(self, query_tensor, key_tensor, queue_tensor):
        """Compare loss with a reference implementation."""
        pos_logits = (query_tensor * key_tensor).sum(dim=1, keepdim=True) / 0.07
        neg_logits = (query_tensor @ queue_tensor.T) / 0.07
        contrastive_acc = (torch.cat([pos_logits, neg_logits], dim=1).argmax(dim=1) == 0).float().mean()

        neg_probs = F.softmax(neg_logits, dim=1)
        neg_entropy = -torch.sum(neg_probs * neg_probs.clamp_min(1e-12).log(), dim=1)
        
        loss = InfoNCELoss(temperature=0.07)
        _, cl_stats = loss(query_tensor, key_tensor, queue_tensor)

        assert torch.allclose(cl_stats["pos_mean"], pos_logits.mean())
        assert torch.allclose(cl_stats["neg_mean"], neg_logits.mean())
        assert torch.allclose(cl_stats["contrastive_acc"], contrastive_acc)
        assert torch.allclose(cl_stats["neg_entropy"], neg_entropy.mean())
    
    def test_forward_small_queue(self, query_tensor, key_tensor, queue_tensor):
        """Test loss with small queue."""
        loss = InfoNCELoss(min_negatives=81)
        loss_value, loss_dict = loss(query_tensor, key_tensor, queue_tensor)
        assert loss_value.shape == ()
        assert loss_value.item() == 0
        assert loss_dict.get("skipped")
