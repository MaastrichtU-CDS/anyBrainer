"""Utility functions for FOMO25 inference."""

import logging
import urllib.request
from pathlib import Path

def download_templates(
        target_dir: Path | str = "templates", 
        template: str | None = None,
) -> None:
    """Download MNI152 templates"""
    base_url = "https://zenodo.org/records/15470657/files/"
    files = [
        "icbm_mni152_t1_09a_asym_bet.nii.gz",
        "icbm_mni152_t2_09a_asym_bet.nii.gz",
    ]
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if template:
        if template not in files:
            logging.error("Unknown template.")
            raise ValueError(f"Unknown template: {template}")
        files_to_download = [template]
    else:
        files_to_download = files

    for fname in files_to_download:
        dest = target_dir / fname
        if dest.exists():
            logging.info(f"{fname} already exists, skipping.")
            continue
        url = base_url + fname
        logging.info(f"Downloading {fname}...")
        try:
            urllib.request.urlretrieve(url, dest)
            logging.info(f"Successfully saved to {dest}")
        except Exception as e:
            logging.info(f"Failed to download {fname}: {e}")