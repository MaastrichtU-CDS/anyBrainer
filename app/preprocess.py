"""Preprocess input images for FOMO25 inference."""

from typing import Literal, cast
from pathlib import Path
import subprocess
import os

import numpy as np
import ants
import yaml
import torch
from monai.utils.type_conversion import convert_to_numpy

Task = Literal["task1", "task2", "task3"]

class PreprocessError(RuntimeError):
    pass

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_nifti(path: Path) -> ants.ANTsImage:
    return ants.image_read(str(path), reorient='RAS') # type: ignore

def _save_nifti(data: ants.ANTsImage, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_file(out_path)

def _has_cmd(cmd: str) -> bool:
    return subprocess.call(["bash", "-lc", f"command -v {cmd} >/dev/null 2>&1"]) == 0

def _get_reg_transforms(
    moving: ants.ANTsImage, 
    template: ants.ANTsImage, 
    transform: str = "Rigid",
) -> tuple[list[str], list[str]]:
    registration = ants.registration(fixed=template, moving=moving, type_of_transform=transform)
    fwd_transform = registration['fwdtransforms']
    inv_transform = registration['invtransforms']
    return fwd_transform, inv_transform

def _apply_transforms(
    moving: ants.ANTsImage,
    target: ants.ANTsImage,
    transformlist: list[str], 
    interp: str = "linear",
) -> ants.ANTsImage:
    return ants.apply_transforms(
        fixed=target, moving=moving, transformlist=transformlist, interpolator=interp
    )

def _apply_mask(image: ants.ANTsImage, mask: ants.ANTsImage) -> ants.ANTsImage:
    return ants.mask_image(image, mask)

def _run_hdbet_cli(image: Path, out_mask: Path, device: str = "cpu", mode: str = "fast", tta: int = 0):
    if not _has_cmd("hd-bet"):
        raise PreprocessError("hd-bet CLI not found on PATH")
    cmd = f"hd-bet -i {image} -o {out_mask} -device {device} -mode {mode} -tta {tta}"
    ret = subprocess.call(["bash", "-lc", cmd])
    if ret != 0 or not out_mask.exists():
        raise PreprocessError("hd-bet CLI failed.")

def _numpy_to_ants(arr, target_img=None):
    """Create an ANTs image from a numpy array with specific properties"""
    new_img = ants.from_numpy(arr)
    if target_img is not None:
        new_img.set_spacing(target_img.spacing) # Preserve meta-data of original image
        new_img.set_origin(target_img.origin)
        new_img.set_direction(target_img.direction)
    return new_img

def _tensor_to_3d_mask_binary(pred, threshold=0.5) -> np.ndarray:
    a = convert_to_numpy(pred, dtype=np.float32)  # preserves shape
    # drop batch
    if a.ndim == 5:  # (B,C,D,H,W)
        a = a[0]
    # drop channel
    if a.ndim == 4:  # (C,D,H,W)
        # expect C==1
        a = a[0]
    # now (D,H,W)
    return (a >= threshold).astype(np.uint8)

def preprocess_inputs(
    inputs: list[Path] | list[Path | None],
    mods: list[str],
    task: Task,
    work_dir: Path| None = None,
    ) -> None:
    """
    Preprocess input images for FOMO25 tasks.

    Steps:
      1) Choose extraction reference (FLAIR for tasks 1/2, T1w for task 3).
      2) Run HD-BET to get brain mask on the reference image.
      3) Register reference to template with ANTs; collect fwd/inv transforms.
      4) Apply brain mask and forward transforms to all inputs (resampling as needed).
      5) Crop to bounding box of the transformed mask.
      6) Save preprocessed NIfTIs to <work_dir>/inputs.

    Args:
        inputs: List of input image paths.
        mods: List of modalities to process (e.g. ["flair", "t1", "t2"]).
        task: Task to preprocess for (one of "task1", "task2", "task3").
        work_dir: Optional target directory for preprocessed images.

    Raises:
        ValueError: If `inputs` and `mods` are not the same length.
        ValueError: If `inputs` and `mods` contain only None elements.
        FileNotFoundError: If any input image does not exist.
        PreprocessError: If hd-bet CLI is not found on PATH.
        PreprocessError: If hd-bet CLI fails.
        PreprocessError: If any input image preprocessing fails.
    """
    if len(inputs) != len(mods):
        raise ValueError("`inputs` and `mods` must be same length.")
    
    # Skip optional inputs/mods
    mods = [m for m, p in zip(mods, inputs) if p is not None]
    inputs = cast(list[Path], [Path(p) for p in inputs if p is not None])

    if len(inputs) == 0:
        raise ValueError("No inputs to preprocess.")

    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    # Setup working dirs
    work_dir = Path(work_dir or os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))
    in_dir = _ensure_dir(work_dir / "inputs")
    mask_dir = _ensure_dir(work_dir / "masks")
    reg_dir = _ensure_dir(work_dir / "reg")

    # Map modalities to input paths
    mod2path: dict[str, Path] = {m.lower(): Path(p) for m, p in zip(mods, inputs)}

    # Choose reference modality for computing brain mask & registration fields
    if task in ("task1", "task2"):
        ref_mod = "flair" if "flair" in mod2path else next(iter(mod2path.keys()))
        tmpl_path = Path("icbm_mni152_t2_09a_asym_bet.nii.gz")
    else:
        ref_mod = "t1" if "t1" in mod2path else next(iter(mod2path.keys()))
        tmpl_path = Path("icbm_mni152_t1_09a_asym_bet.nii.gz")
    
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Template not found: {tmpl_path}")

    ref_path = mod2path[ref_mod]
    ref_img = _load_nifti(ref_path)

    # Skull-stripping via HD-BET
    mask_path = mask_dir / "brain_mask.nii.gz"
    _run_hdbet_cli(ref_path, mask_path)
    mask_img = _load_nifti(mask_path)

    # Registration to template
    tmpl_img = _load_nifti(tmpl_path)
    fwd, inv = _get_reg_transforms(_apply_mask(ref_img, mask_img), tmpl_img)
    ants.read_transform(fwd[0], precision='float')
    ants.write_transform(ants.read_transform(fwd[0], precision='float'), str(reg_dir / "fwd.mat"))
    ants.write_transform(ants.read_transform(inv[0], precision='float'), str(reg_dir / "inv.mat"))

    # Get crop slices
    mask_img_reg = _apply_transforms(mask_img, tmpl_img, fwd)

    # Preprocess all inputs
    for img_path in inputs:
        try:
            img = _load_nifti(img_path)
            img_reg = _apply_transforms(img, tmpl_img, fwd)
            _save_nifti(_numpy_to_ants(img_reg), in_dir / img_path.name)
        except Exception as e:
            raise PreprocessError(f"Failed to preprocess {img_path}: {e}")

def revert_preprocess(
    pred: torch.Tensor,
    orig_img: ants.ANTsImage,
    work_dir: Path| None = None,
) -> ants.ANTsImage:
    """
    Revert preprocessing on predicted segmentation masks for FOMO25 tasks.

    Assumes that following `predict_transforms` are already reversed:
    - Padding
    - Cropping
    - Resampling
    - Orientation (flipping)

    For now assumes that it is used only for task 2 that contains FLAIR (template is T2w).
    
    Steps: 
    1) Convert to uint8 numpy array
    2) Load fwd/inv transforms from reg.mat
    3) Apply transforms to pred

    Args:
        pred: Predicted segmentation mask as torch.Tensor.
        orig_img: Original image as ants.ANTsImage.
        work_dir: Optional target directory for preprocessed images.

    Returns:
        Reversed segmentation mask as ants.ANTsImage.
    """
    work_dir = Path(work_dir or os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))

    inv_tansform = work_dir / "reg" / "inv.mat"
    if not inv_tansform.exists():
        raise FileNotFoundError(f"Inverse transform not found: {inv_tansform}")
    
    tmpl_path = Path("icbm_mni152_t2_09a_asym_bet.nii.gz")
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Template not found: {tmpl_path}")
    tmpl_img = _load_nifti(tmpl_path)

    pred_arr = _tensor_to_3d_mask_binary(pred)
    pred_img = _numpy_to_ants(pred_arr, tmpl_img)

    return _apply_transforms(pred_img, orig_img, [str(inv_tansform)])
    