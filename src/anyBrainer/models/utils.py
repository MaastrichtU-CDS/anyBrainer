"""Utility functions for creating and running models."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.inferers.inferer import SlidingWindowInferer

MODALITY_LABELS = ["t1", "t2", "flair", "dwi", "adc", "swi", "other"]
MODALITY_TO_INDEX = {modality: idx for idx, modality in enumerate(MODALITY_LABELS)}

logger = logging.getLogger(__name__)

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    else:
        msg = f"Unsupported activation: {name}"
        logger.error(msg)
        raise ValueError(msg)

def modality_to_onehot(batch: dict, key: str, device: torch.device) -> torch.Tensor:
    """
    Convert a batch of modality strings to one-hot encoded tensors.

    Args:
        batch (dict): Dictionary containing the modality list under `key`.
        key (str): The key to access modality names (must be a list).
        device (torch.device): Target device for the output tensor.

    Returns:
        torch.Tensor: Tensor of shape (B, 7) containing one-hot vectors.
    """
    modalities = batch.get(key, None)
    if modalities is None:
        msg = f"Key '{key}' not found in batch."
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(modalities, list):
        msg = (f"Expected a list of modalities under key '{key}', "
               f"but got {type(modalities)}.")
        logger.error(msg)
        raise TypeError(msg)

    one_hots = torch.zeros((len(modalities), len(MODALITY_LABELS)),
                           dtype=torch.float32, device=device)

    for i, modality in enumerate(modalities):
        index = MODALITY_TO_INDEX.get(modality.lower(), MODALITY_TO_INDEX["other"])
        one_hots[i, index] = 1.0

    return one_hots

def top1_accuracy(logits: torch.Tensor, targets_one_hot: torch.Tensor) -> torch.Tensor:
    """Compute top-1 accuracy."""
    targets_idx = targets_one_hot.argmax(dim=1)
    preds = logits.argmax(dim=1)
    return (preds == targets_idx).float().mean()

def get_inferer_from_roi_size(
    roi_size: tuple[int, int, int] | int, 
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    **kwargs,
) -> SlidingWindowInferer:
    """Get a sliding window inferer from a ROI size."""
    if isinstance(roi_size, int):
        roi_size = (roi_size, roi_size, roi_size)
        
    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        **kwargs,
    )

def compute_cl_stats(
    logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute contrastive learning statistics.

    Args:
        logits: (B, 1 + K) - logits from the model (1 positive, K negatives)
        temperature: float - temperature for the softmax function

    Returns:
        dict[str, torch.Tensor] - dictionary with the statistics
    """
    with torch.no_grad():
        # Mean negative similarity
        finite_neg = torch.isfinite(logits[:, 1:]) # Only consider finite negatives (after masking)
        neg_mean = (
            logits[:, 1:][finite_neg].mean()
            if finite_neg.any()
            else torch.tensor(float("nan"), device=logits.device)
        )
        # Contrastive accuracy
        contrastive_acc = (logits.argmax(dim=1) == 0).float().mean()

        # Negative entropy
        neg_probs = F.softmax(logits[:, 1:], dim=1)
        neg_entropy = -torch.sum(neg_probs * neg_probs.clamp_min(1e-12).log(), dim=1)

    return {
        "pos_mean": logits[:, 0].mean().detach(),
        "neg_mean": neg_mean.detach(),
        "contrastive_acc": contrastive_acc.detach(),
        "neg_entropy": neg_entropy.mean().detach(),
    }

def count_model_params(model: nn.Module, trainable: bool = False):
    """Count model parameters."""
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def summarize_model_params(model: nn.Module) -> str:
    """Show summary of all model parameters and buffers."""
    out_msg = "#Model parameters#\n"
    for name, param in model.named_parameters():
        out_msg += f"{name:60s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}\n"
        
    out_msg += f"\n#Model buffers#\n"
    for name, b in model.named_buffers():
        out_msg += f"{name:55s} {tuple(b.shape)}\n"
        
    out_msg += f"\n#Model summary#\n"
    out_msg += f"Total parameters: {count_model_params(model)}\n"
    out_msg += f"Trainable parameters: {count_model_params(model, trainable=True)}\n"
    out_msg += f"Non-trainable parameters: {count_model_params(model, trainable=False)}\n"
    
    return out_msg

def get_optimizer_lr(optimizers: list[optim.Optimizer]) -> dict[str, float]:
    """Get optimizer learning rates."""
    return {
        f"train/lr/opt{i}_group{j}": group["lr"]
        for i, opt in enumerate(optimizers)
        for j, group in enumerate(opt.param_groups)
    }

def get_total_grad_norm(model: nn.Module) -> torch.Tensor:
    """Get total gradient norm."""
    total_norm = torch.norm(
        torch.stack([
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]), p=2,
    )
    return total_norm

def log_gradients_norm(model: nn.Module) -> None:
    """Log gradients norm."""
    total_norm = get_total_grad_norm(model)
    logger.info(f"Total gradient norm: {total_norm}")