"""Utility functions for datamodule operations."""

import logging
from pathlib import Path
import re
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

def parse_filename_nested_nifti(file_path: Path | str) -> dict:
    """
    Parse filename with pattern: root/sub_x/ses_y/ModalityName_CountIfMoreThanOne.nii.gz
    """
    file_path = Path(file_path)

    file_name = file_path.name
    modality = file_name.split('_')[0]
    ses_dir = file_path.parent
    sub_dir = ses_dir.parent

    return {
        'sub_id': sub_dir.name,
        'ses_id': ses_dir.name,
        'modality': modality,
        'file_name': file_name
    }

def parse_filename_flat_npy(file_path: Path | str) -> Optional[Dict]:
    """
    Parse filename with pattern: sub_x_ses_y_ModalityName_CountIfMoreThanOne.npy
    
    Returns:
        Dict with keys: sub_id, ses_id, modality, count, filepath
    """
    # Remove .npy extension
    base_name = Path(file_path).name.replace('.npy', '')
    
    # Pattern: sub_(\d+)_ses_(\d+)_(.+)
    pattern = r'sub_(\d+)_ses_(\d+)_(.+)'
    match = re.match(pattern, base_name)
    
    if not match:
        logger.warning(f"Could not parse filename in: {file_path}")
        return None
        
    sub_id = match.group(1)
    ses_id = match.group(2)
    modality_part = match.group(3)
    
    # Check if modality has count (ends with _number)
    count_pattern = r'(.+)_(\d+)$'
    count_match = re.match(count_pattern, modality_part)
    
    if count_match:
        modality = count_match.group(1)
        count = int(count_match.group(2))
        modality_suffix = f"{modality}_{count}"
    else:
        modality = modality_part
        count = 1
        modality_suffix = modality
        
    return {
        'sub_id': sub_id,
        'ses_id': ses_id,
        'modality': modality,
        'count': count,
        'modality_suffix': modality_suffix,
        'filepath': str(file_path)
    }

def check_flat_npy_data_dir(data_dir: Path | str) -> None:
    """
    Check if data_dir contains the .npy files and log basic statistics.
    Assumes that the data is in a flat directory structure, with .npy files 
    saved as sub_x_ses_y_ModalityName_CountIfMoreThanOne.npy.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    logger.info(f"Collecting data from {data_dir}")
    
    npy_files = list(data_path.glob("*.npy"))
    if len(npy_files) == 0:
        logger.error(f"No .npy files found in {data_dir}")
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
        
    logger.info(f"Found {len(npy_files)} .npy files in {data_dir}")
    
    # Parse filenames to get basic statistics
    subjects = set()
    sessions = set()
    modalities = set()
    
    for file_path in npy_files:
        metadata = parse_filename_flat_npy(file_path)
        if metadata:
            subjects.add(metadata['sub_id'])
            sessions.add(f"{metadata['sub_id']}_ses_{metadata['ses_id']}")
            modalities.add(metadata['modality'])
    
    logger.info(f"Dataset contains {len(subjects)} subjects, "
                f"{len(sessions)} sessions, {len(modalities)} modalities")

def trivial_check_nested_nifti_dataset(data_dir: Path | str) -> None:
    """
    Trivial check for nested nifti dataset.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
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