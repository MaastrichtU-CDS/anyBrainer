"""Utility functions for data operations."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODALITY_LABELS = ["t1", "t2", "flair", "dwi", "other"]
MODALITY_TO_INDEX = {modality: idx for idx, modality in enumerate(MODALITY_LABELS)}

def modality_to_idx(
    batch: dict, 
    key: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert a list of modality strings in `batch[key]` to a 1‑D tensor of
    class indices suitable for nn.CrossEntropyLoss.

    If a modality string is unknown, it falls back to the index for "other".

    Returns:
        torch.Tensor: Tensor of shape (B, n_modalities) containing one-hot vectors.
    """
    modalities = batch.get(key)
    if modalities is None:
        raise ValueError(f"Key '{key}' not found in batch.")
    if not isinstance(modalities, list):
        raise TypeError(f"Expected list of modalities under key '{key}', got {type(modalities)}")

    idx_list = [
        MODALITY_TO_INDEX.get(str(m).lower(), MODALITY_TO_INDEX["other"])
        for m in modalities
    ]
    return torch.tensor(idx_list, dtype=torch.long, device=device)

def modality_to_onehot(
    batch: dict, 
    key: str, 
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert a batch of modality strings to one-hot encoded tensors.

    Args:
        batch (dict): Dictionary containing the modality list under `key`.
        key (str): The key to access modality names (must be a list).
        device (torch.device): Target device for the output tensor.

    Returns:
        torch.Tensor: Tensor of shape (B, n_modalities) containing one-hot vectors.
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
    
def split_data_by_subjects(
    data_list: list[dict], 
    train_val_test_split: tuple = (0.7, 0.15, 0.15), 
    random_state: np.random.RandomState | None = None,
    seed: int | None = None,
) -> tuple:
    """
    Split data into train/val/test based on subjects for proper separation.
    Assumes that data_list is a list of dictionaries with 'sub_id' key.
    """
    # Normalize train_val_test_split to sum to 1
    train_val_test_split = tuple(
        np.array(train_val_test_split) / sum(train_val_test_split)
    )

    if random_state is not None:
        R = random_state
    elif seed is not None:
        R = np.random.RandomState(seed)
    else:
        R = np.random.RandomState()

    _, mt_state, pos, *_ = R.get_state()
    logger.info(f"Splitting data into train/val/test based on subjects for "
                f"masked autoencoder with current state: "
                f"{pos:3d}, {mt_state[:5]}") # pyright: ignore[reportArgumentType]
    
    # Get unique subjects
    subjects = sorted({item['sub_id'] for item in data_list})
    
    # Shuffle subjects for random split
    R.shuffle(subjects)
    
    # Calculate split indices
    train_end = int(len(subjects) * train_val_test_split[0])
    val_end = train_end + int(len(subjects) * train_val_test_split[1])
    
    train_subjects = set(subjects[:train_end])
    val_subjects = set(subjects[train_end:val_end])
    test_subjects = set(subjects[val_end:])
    
    # Split data based on subject assignment
    train_data = [item for item in data_list if item['sub_id'] in train_subjects]
    val_data = [item for item in data_list if item['sub_id'] in val_subjects]
    test_data = [item for item in data_list if item['sub_id'] in test_subjects]
    
    logger.info(f"Data split - Train: {len(train_data)}, "
                f"Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data