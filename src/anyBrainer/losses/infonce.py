"""Custom loss functions."""

__all__ = [
    "InfoNCELoss",
]

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from anyBrainer.losses.utils import compute_cl_stats

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with optional top-k hard negative selection and
    optional (l_pos, l_neg) post-processing hook.

    Args:
        temperature: Softmax temperature.
        top_k_negatives: If set, keep only the top-k highest similarity negatives per sample.
        postprocess_fn: Optional fn(l_pos, l_neg) -> (l_pos, l_neg) for custom filtering or scaling.
        normalize: L2-normalize q and k (assumes queue already normalized when enqueued).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        top_k_negatives: int | None = None,
        postprocess_fn: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        normalize: bool = False,
        cross_entropy_args: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k_negatives = top_k_negatives
        self.postprocess_fn = postprocess_fn
        self.normalize = normalize
        self.cross_entropy_args = cross_entropy_args or {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        queue: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            q: (B, D) - query embeddings
            k: (B, D) - positive key embeddings (same samples as q)
            queue: (K, D) - bank of negative keys, assumed normalized if normalize=True
        Returns:
            loss: scalar tensor
            stats: dict with metrics like contrastive accuracy, pos/neg means
        """
        B, D = q.shape

        if k.shape != (B, D):
            raise ValueError(f"k.shape {k.shape} != {(B, D)}")

        if queue.dim() != 2 or queue.shape[1] != D:
            raise ValueError(f"queue.shape {queue.shape} expected (K, {D})")

        K = queue.shape[0]

        if self.normalize:
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            # Assume queue already normalized externally.

        # Positive logits (B,1)
        l_pos = (q * k).sum(dim=1, keepdim=True)

        # Negative logits (B,K)
        with torch.no_grad():
            negatives = queue.detach()  # (K,D)
        l_neg = q @ negatives.T        # (B,K)

        # Top-k hard negative selection
        if self.top_k_negatives is not None:
            M = self.top_k_negatives
            if M > K or M <= 0:
                msg = "top_k_negatives must not exceed queue size and be positive."
                logger.error(msg)
                raise ValueError(msg)
            # Indices of top-M similarities
            _, idx = torch.topk(l_neg, k=M, dim=1)
            mask = torch.zeros_like(l_neg, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            l_neg = l_neg.masked_fill(~mask, float('-inf'))

        # Optional custom post-processing
        if self.postprocess_fn is not None:
            l_pos, l_neg = self.postprocess_fn(l_pos, l_neg)

        # Concatenate positive in front
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels, **self.cross_entropy_args)

        return loss, compute_cl_stats(logits)