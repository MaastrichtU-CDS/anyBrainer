"""Inferers for classification tasks."""

__all__ = [
    "SlidingWindowClassificationInferer"
]

import logging
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from monai.inferers.inferer import (
    Inferer,
)
from monai.data.utils import dense_patch_slices

from anyBrainer.core.inferers.utils import (
    ensure_tuple_of_length,
    get_patch_gaussian_weight,
)
from anyBrainer.registry import register
from anyBrainer.registry import RegistryKind as RK

logger = logging.getLogger(__name__)


@register(RK.INFERER)
class SlidingWindowClassificationInferer(Inferer):
    """
    Collects classification predictions from a sliding window and 
    optionally assigns a single label. 

    Predictions are then aggregated using a weighted average, based
    on the gaussian distance from the center of the image. 
    """
    def __init__(
        self, 
        patch_size: int | Sequence[int],
        overlap: float | Sequence[float],
        padding_mode: str = "constant",
        aggregation_mode: Literal["weighted", "mean", "majority", "none"] = "weighted",
        apply_activation: bool = True,
    ):
        """
        Args:
        - patch_size: The size of the patches to extract from the image.
        - overlap: The overlap between patches.
        - padding_mode: The mode to use for padding the image.
        - aggregation_mode: The mode to use for aggregating the predictions.
        - apply_activation: Whether to apply activation to the predictions;
            softmax for multiclass, sigmoid for binary.
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.padding_mode = padding_mode
        if aggregation_mode not in ["weighted", "mean", "majority", "none"]:
            msg = f"Invalid aggregation mode: {aggregation_mode}"
            raise ValueError(msg)
        self.aggregation_mode = aggregation_mode
        self.apply_activation = apply_activation

        logger.info(f"[{self.__class__.__name__}] Initialised with "
                    f"patch_size={patch_size}, overlap={overlap}, "
                    f"padding_mode={padding_mode}, "
                    f"aggregation_mode={aggregation_mode}, "
                    f"apply_activation={apply_activation}")
    
    def __call__(self, 
        inputs: torch.Tensor, 
        network: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Args:
        - inputs: The input tensor.
        - network: The network to use for inference.
        """
        try:
            B, C, *spatial = inputs.shape
        except ValueError:
            logger.exception(f"Inputs must have at least B, C, *spatial dimensions, "
                             f"got {inputs.shape}")
            raise
        
        device = inputs.device
        n_dim = len(spatial)
        patch_size = ensure_tuple_of_length(self.patch_size, n_dim)
        overlap = ensure_tuple_of_length(self.overlap, n_dim)

        # Convert relative overlap to absolute overlap for dense_patch_slices()
        scan_interval = tuple(max(1, int(round(ps * (1.0 - ov))))
                              for ps, ov in zip(patch_size, overlap))

        # Pad input if necessary
        pad_needed = [(max(p - s, 0), 0) for p, s in zip(reversed(patch_size), reversed(spatial))]
        if any(pad_left > 0 for pad_left, _ in pad_needed):
            flat_pad = [item for pair in pad_needed for item in pair]
            inputs = F.pad(inputs, flat_pad, mode=self.padding_mode)
            spatial = inputs.shape[2:]
        
        # Get patch slices and weights
        slices = dense_patch_slices(spatial, patch_size, scan_interval)
        img_center = [(s - 1) / 2.0 for s in spatial]
        sigma = [ps / 2.0 for ps in patch_size]
        weights: list[float] = []

        if self.aggregation_mode == "weighted":
            for slc in slices:
                start_idxs = [s.start for s in slc]
                center = [start + ps / 2.0 for start, ps in zip(start_idxs, patch_size)]
                weights.append(get_patch_gaussian_weight(center, img_center, sigma))
        elif self.aggregation_mode == "mean":
            weights = [1.0] * len(slices)

        weight_tensor = torch.tensor(weights, device=device) # (N_patches,)
        weight_tensor = weight_tensor / weight_tensor.sum() # normalise

        # Get predictions per patch
        prob_lists: list[torch.Tensor] = [] # (N_patches, B, num_classes)
        for slc in slices:
            patch = inputs[(slice(None), slice(None)) + slc] # (B, C, *patch_size)
            logits = network(patch) # (B, num_classes)
            if self.apply_activation:
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits) # (B, 1)
                else:
                    probs = torch.softmax(logits, dim=1) # (B, num_classes)
            else:
                probs = logits # raw logits
            prob_lists.append(probs)
        
        probs_all = torch.stack(prob_lists, dim=0).permute(1, 0, 2) # (B, N_patches, num_classes)

        # Aggregate predictions
        num_classes = probs_all.shape[-1]

        if self.aggregation_mode == "none":
            return probs_all

        if self.aggregation_mode == "majority":
            if num_classes == 1:
                # Binary: vote on sigmoid output ≥ 0.5
                patch_preds = (probs_all >= 0.5).long().squeeze(-1)  # (B, N_patches)
                majority_vote = torch.mode(patch_preds, dim=1).values  # (B,)
                return majority_vote.unsqueeze(1).float()  # (B, 1)
            else:
                # Multi-class: vote on argmax
                patch_preds = probs_all.argmax(dim=2)  # (B, N_patches)
                majority_vote = torch.mode(patch_preds, dim=1).values  # (B,)
                return F.one_hot(majority_vote, num_classes).float()  # (B, num_classes)

        return (probs_all * weight_tensor.view(1, -1, 1)).sum(dim=1) # (B, n_classes)