"""
Mock data generation utilities for testing.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

def create_mock_brain_volume(
    shape: Tuple[int, int, int, int] = (1, 64, 64, 64),
    background_value: float = 0.0,
    brain_value_range: Tuple[float, float] = (100.0, 1000.0),
    add_noise: bool = True
) -> np.ndarray:
    """
    Create a mock 3D brain volume with realistic characteristics.
    
    Args:
        shape: (C, H, W, D) shape of the volume
        background_value: Value for background voxels
        brain_value_range: (min, max) range for brain tissue values
        add_noise: Whether to add Gaussian noise
    
    Returns:
        Mock brain volume as numpy array
    """
    c, h, w, d = shape
    volume = np.full(shape, background_value, dtype=np.float32)
    
    # Create brain tissue in center region
    brain_margin = 8
    brain_region = (
        slice(None),  # All channels
        slice(brain_margin, h - brain_margin),
        slice(brain_margin, w - brain_margin), 
        slice(brain_margin, d - brain_margin)
    )
    
    # Fill brain region with random values in realistic range
    brain_size = (
        c,
        h - 2 * brain_margin,
        w - 2 * brain_margin,
        d - 2 * brain_margin
    )
    
    brain_values = np.random.uniform(
        brain_value_range[0], 
        brain_value_range[1], 
        brain_size
    )
    
    if add_noise:
        noise = np.random.normal(0, brain_value_range[1] * 0.05, brain_size)
        brain_values += noise
    
    volume[brain_region] = brain_values
    
    return volume

def create_mock_dataset_files(
    data_dir: Path,
    n_subjects: int = 3,
    sessions_per_subject: int = 2,
    modalities: List[str] = ["T1w", "T2w", "FLAIR"],
    volume_shape: Tuple[int, int, int, int] = (1, 64, 64, 64)
) -> List[str]:
    """
    Create a complete mock dataset with proper filename conventions.
    
    Args:
        data_dir: Directory to create files in
        n_subjects: Number of subjects
        sessions_per_subject: Number of sessions per subject
        modalities: List of modality names
        volume_shape: Shape of each volume
    
    Returns:
        List of created filenames
    """
    created_files = []
    
    for sub_id in range(1, n_subjects + 1):
        for ses_id in range(1, sessions_per_subject + 1):
            for modality in modalities:
                # Create filename
                filename = f"sub_{sub_id:03d}_ses_{ses_id:03d}_{modality}.npy"
                filepath = data_dir / filename
                
                # Create mock data with slight variations per modality
                if modality == "T1w":
                    brain_range = (200, 1000)
                elif modality == "T2w":
                    brain_range = (100, 800)
                elif modality == "FLAIR":
                    brain_range = (50, 600)
                else:
                    brain_range = (100, 800)
                
                volume = create_mock_brain_volume(
                    shape=volume_shape,
                    brain_value_range=brain_range
                )
                
                np.save(filepath, volume)
                created_files.append(filename)
    
    return created_files