"""Utility functions for FOMO25 inference."""

import logging
import urllib.request
from pathlib import Path

import torch

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

def get_pads_from_bbox(
    crop_slices: tuple[slice, slice, slice],
    orig_shape_xyz: tuple[int, int, int],
) -> dict[str, tuple[int, int]]:
    """
    Returns:
      dict with target pads {"x": (left, right), "y": (left, right), "z": (left, right)}
    """
    sx, sy, sz = crop_slices
    X, Y, Z = orig_shape_xyz

    def lr(s: slice, dim: int) -> tuple[int, int]:
        start = 0 if s.start is None else int(s.start)
        stop  = dim if s.stop  is None else int(s.stop)
        return start, dim - stop

    px = lr(sx, X)  # (left, right) along X
    py = lr(sy, Y)  # (left, right) along Y
    pz = lr(sz, Z)  # (left, right) along Z

    return {"x": px, "y": py, "z": pz}

def write_probability(output_path: Path, prob: float):
    """Write single probability (float) to a .txt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{float(prob):.6f}\n")  # six decimal places, trailing newline

def get_device() -> torch.device:
    # allow override via env/flag later if you want
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    # move just the image tensor; meta dicts/lists stay on CPU
    if "img" in batch and isinstance(batch["img"], torch.Tensor):
        batch["img"] = batch["img"].to(device, non_blocking=True)
    return batch