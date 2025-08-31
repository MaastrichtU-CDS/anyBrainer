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

from anyBrainer.core.inferers.utils import (
    get_patch_gaussian_weight,
)
from anyBrainer.core.utils import (
    ensure_tuple_dim,
    noisy_or_bag_logits,
    lse_bag_logits_from_instance_logits,
    topk_mean_bag_logits,
)
from anyBrainer.core.transforms.unit_transforms import (
    SlidingWindowPatch,
)
from anyBrainer.registry import register
from anyBrainer.registry import RegistryKind as RK

logger = logging.getLogger(__name__)


@register(RK.INFERER)
class SlidingWindowClassificationInferer(Inferer):
    """
    Collects classification predictions from a sliding window and 
    optionally assigns a single label. 

    Predictions are then aggregated using standard voting schemes or MIL-based aggregation.
    - `weighted`: weighted average, based on the gaussian distance from the center of the image.
    - `mean`: mean of the predictions.
    - `majority`: majority vote.
    - `noisy_or`: noisy OR aggregation.
    - `none`: no aggregation.
    """
    def __init__(
        self, 
        patch_size: int | Sequence[int],
        overlap: float | Sequence[float] | None = None,
        n_patches: int | Sequence[int] | None = None,
        spatial_dims: int = 3,
        padding_mode: str = "constant",
        aggregation_mode: Literal["weighted", "mean", "majority", "none", "noisy_or", "lse", "topk"] = "noisy_or",
        apply_activation: bool = False,
        *,
        topk: int = 4,
        tau: float = 2.0,
    ):
        """
        Args:
        - patch_size: The size of the patches to extract from the image.
        - overlap: The overlap between patches.
        - n_patches: The number of patches to extract from the image; 
            if provided, `overlap` is ignored.
        - padding_mode: The mode to use for padding the image.
        - aggregation_mode: The mode to use for aggregating the predictions.
        - apply_activation: Whether to apply activation to the predictions;
            softmax for multiclass, sigmoid for binary.
        """
        self.spatial_dims = spatial_dims
        self.slice_extractor = SlidingWindowPatch(
            patch_size=patch_size,
            overlap=overlap,
            n_patches=n_patches,
            padding_mode=padding_mode,
            spatial_dims=spatial_dims,
        )
        if aggregation_mode not in ["weighted", "mean", "majority", "none", "noisy_or", "lse", "topk"]:
            msg = f"Invalid aggregation mode: {aggregation_mode}"
            raise ValueError(msg)
        self.aggregation_mode = aggregation_mode
        self.apply_activation = apply_activation
        self.topk = int(topk)
        self.tau = float(tau)

        logger.info(f"[{self.__class__.__name__}] Initialised with "
                    f"patch_size={patch_size}, overlap={overlap}, "
                    f"n_patches={n_patches}, "
                    f"padding_mode={padding_mode}, "
                    f"aggregation_mode={aggregation_mode}, "
                    f"apply_activation={apply_activation}, "
                    f"topk={topk}, tau={tau}")
    
    def __call__(self, 
        inputs: torch.Tensor, 
        network: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Args:
        - inputs: The input tensor.
        - network: The network to use for inference.
        """
        if inputs.ndim != self.spatial_dims + 2: # (B, C, *spatial_dims)
            msg = f"Inputs must have {self.spatial_dims + 2} dimensions, " \
                  f"got {inputs.ndim}"
            logger.error(msg)
            raise ValueError(msg)
        
        B, C, *spatial = inputs.shape
        
        device = inputs.device

        # Get slices for patches
        self.slice_extractor(inputs)
        slices = self.slice_extractor.slices
        P = len(slices)

        logits_list: list[torch.Tensor] = []
        for slc in slices:
            patch = inputs[(slice(None), slice(None)) + slc]  # (B, C, *patch_size)
            logits = network(patch)  # (B, num_classes)
            logits_list.append(logits)
        logits_all = torch.stack(logits_list, dim=1) # (B, P, n_classes)

        # Aggregate predictions
        if self.aggregation_mode == "topk":
            z = logits_all.transpose(1, 2) # (B, n_classes, P)
            logits = topk_mean_bag_logits(z, dims=(-1,), k=self.topk) # (B, n_classes)

        elif self.aggregation_mode == "lse":
            z = logits_all.transpose(1, 2) # (B, n_classes, P)
            logits = lse_bag_logits_from_instance_logits(z, dims=(-1,), tau=self.tau)

        elif self.aggregation_mode == "noisy_or":
            z = logits_all.transpose(1, 2) # (B, n_classes, P)
            logits = noisy_or_bag_logits(z, dims=(-1,))
            
        elif self.aggregation_mode == "mean":
            logits = logits_all.mean(dim=1)

        elif self.aggregation_mode == "weighted":
            img_center = [(s - 1) / 2.0 for s in spatial]
            sigma = [ps / 2.0 for ps in self.slice_extractor.patch_size]
            weights = []
            for slc in slices:
                start_idxs = [s.start for s in slc]
                center = [start + ps / 2.0 for start, ps in zip(start_idxs, self.slice_extractor.patch_size)]
                weights.append(get_patch_gaussian_weight(center, img_center, sigma))
            w = torch.as_tensor(weights, device=device, dtype=logits_all.dtype) # (P,)
            w = w / (w.sum() + 1e-12)
            logits = (logits_all * w.view(1, -1, 1)).sum(dim=1) # (B, n_classes)

        elif self.aggregation_mode == "majority":
            probs_all = (torch.sigmoid(logits_all) if logits_all.shape[1] == 1
                         else torch.softmax(logits_all, dim=-1))
            preds = probs_all.argmax(dim=2) # (B, P)
            votes = torch.mode(preds, dim=1).values # (B,)
            return F.one_hot(votes, probs_all.shape[-1]).float() # (B, n_classes)

        elif self.aggregation_mode == "none":
            # return all per-patch logits/probs
            if self.apply_activation:
                if logits_all.shape[-1] == 1:
                    return torch.sigmoid(logits_all) # (B, P, 1)
                return torch.softmax(logits_all, dim=-1) # (B, P, n_classes)
            return logits_all # (B, P, n_classes)

        else:
            msg = f"Unknown aggregation mode: {self.aggregation_mode}"
            logger.error(msg)
            raise ValueError(msg)

        # Final activation
        if self.apply_activation and self.aggregation_mode not in ("majority"):
            if logits.shape[1] == 1:
                return torch.sigmoid(logits) # (B, 1)
            return torch.softmax(logits, dim=1) # (B, n_classes)
        return logits # (B, n_classes)