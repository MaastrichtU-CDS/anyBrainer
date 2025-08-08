"""Utility functions for data operations."""

import logging
from pathlib import Path
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Literal, cast, Sequence

from anyBrainer.registry import get, RegistryKind as RK
from anyBrainer.factories import UnitFactory

logger = logging.getLogger(__name__)

def parse_filename_nested_nifti(file_path: Path | str, ext: str = ".npy") -> dict:
    """
    Parse filename with pattern: root/sub_x/ses_y/ModalityName_CountIfMoreThanOne.npy
    """
    file_path = Path(file_path)

    file_name = file_path.name
    modality = file_name.split(ext)[0].split('_')[0]
    ses_dir = file_path.parent
    sub_dir = ses_dir.parent

    return {
        'sub_id': sub_dir.name,
        'ses_id': ses_dir.name,
        'modality': modality,
        'file_name': file_path
    }

def parse_filename_flat_npy(file_path: Path | str) -> dict | None:
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

def check_data_dir_exists(data_dir: Path | str) -> None:
    """
    Trivial check for nested nifti dataset.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

def group_data(
    group_by: Literal["modality","subject", "session", "img"], 
    data_list: list[Path],
) -> dict[str, Any]:
    """
    Group data by a given key.

    Assumes that filename contains: sub_x_ses_y_ModalityName_CountIfMoreThanOne.npy
    """
    grouped_data = defaultdict(list)
    subjects = set()
    sessions = set()
    modality_counts = Counter()

    # Group by session
    for file_path in data_list:
        metadata = parse_filename_nested_nifti(file_path)
        subject = metadata['sub_id']
        session = f"{metadata['sub_id']}_{metadata['ses_id']}"
        modality = metadata['modality']
        
        subjects.add(subject)
        sessions.add(session)
        modality_counts[modality] += 1

        if group_by == "modality":
            grouped_data[modality].append(metadata)
        elif group_by == "subject":
            grouped_data[subject].append(metadata)
        elif group_by == "session":
            grouped_data[session].append(metadata)
        elif group_by == "img":
            grouped_data[file_path].append(metadata)

    return {
        "grouped_data": grouped_data,
        "subjects": subjects,
        "sessions": sessions,
        "modality_counts": modality_counts,
    }

def get_summary_msg(
    subjects: set, 
    sessions: set, 
    modality_counts: Counter,
) -> str:
    """
    Get summary message for list of data creation.
    """
    msg = f"#### Data Collection Completed ####\nSummary:"
    msg += f"\n  - {len(subjects)} subjects"
    msg += f"\n  - {len(sessions)} sessions"
    msg += f"\n  - {sum(modality_counts.values())} scans"
    msg += f"\n  - Modality distribution:"
    
    for mod, count in modality_counts.items():
        msg += (f"\n    - {mod}: {count} files "
                f"({count/sum(modality_counts.values())*100:.2f}%)")

    return msg

def get_summary_msg_w_labels(
    subjects: set, 
    sessions: set, 
    modality_counts: Counter,
    label_counts: Counter,
) -> str:
    """
    Get summary message for list of data creation.
    """
    msg = f"\n#### Data Collection Completed ####\nSummary:"
    msg += f"\n  - {len(subjects)} subjects"
    msg += f"\n  - {len(sessions)} sessions"
    msg += f"\n  - {sum(modality_counts.values())} scans"
    msg += f"\n  - {len(label_counts)} labels"
    msg += f"\n  - Label distribution:"
    
    for label, count in label_counts.items():
        msg += (f"\n    - {label}: {count} files "
                f"({count/sum(label_counts.values())*100:.2f}%)")

    return msg

def resolve_transform(
    transform: dict[str, Any] | str | list[Callable] | None,
) -> list[Callable] | None:
    """Get transform list from config."""
    if isinstance(transform, dict):
        return UnitFactory.get_transformslist_from_kwargs(transform)

    if isinstance(transform, str):
        return cast(Callable, get(RK.TRANSFORM, transform))()
    
    return transform

def resolve_fn(
    fn: Callable | str | None,
) -> Callable | None:
    """Get function from config."""
    if isinstance(fn, str):
        return cast(Callable, get(RK.UTIL, fn))
    
    return fn

def read_label_from_txt(
    file_path: Path | str, 
    expected_labels: list[str] = ['0', '1'],
    strict: bool = True,
) -> int | None:
    """Read label from txt file."""
    file_path = Path(file_path)
    content = file_path.read_text().strip()

    if content not in expected_labels:
        msg = (f"Unexpected label '{content}' in {file_path}. "
               f"Expected one of {expected_labels}.")
        if strict:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return None

    return int(content)