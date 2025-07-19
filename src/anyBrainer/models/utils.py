"""Utility functions for creating and running models."""

import logging

import torch
import torch.nn as nn

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