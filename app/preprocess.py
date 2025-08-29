"""Preprocess input images for FOMO25 inference."""

from typing import Literal, cast
from pathlib import Path
import subprocess
import os
import shutil
import logging
import uuid

import numpy as np
import ants
import torch
from monai.utils.type_conversion import convert_to_numpy

Task = Literal["task1", "task2", "task3"]

class PreprocessError(RuntimeError):
    pass

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_nifti(path: Path) -> ants.ANTsImage:
    return ants.image_read(str(path))

def _save_nifti(data: ants.ANTsImage, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_file(out_path)

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

def _resolve_hdbet() -> str:
    p = os.environ.get("HDBET_BIN")
    if p and Path(p).exists():
        return p
    for cand in ("/usr/local/bin/hd-bet", "/usr/bin/hd-bet"):
        if Path(cand).exists():
            return cand
    found = shutil.which("hd-bet")
    if found:
        return found
    raise PreprocessError("hd-bet CLI not found in container PATH.")

def _ensure_hdbet_weights() -> Path:
    # hd-bet v2 looks in ~/hd-bet_params/release_2.0.0/...
    home = os.environ.get("HDBET_HOME", "/opt/hd-bet-home")
    root = Path(home) / "hd-bet_params" / "release_2.0.0"
    ck = root / "fold_all" / "checkpoint_final.pth"
    if not ck.exists():
        raise PreprocessError(
            f"HD-BET weights not found: {ck}. The container is offline; "
            f"ship weights under {root}."
        )
    return root

def _run_hdbet_cli(image: Path, out_mask: Path, device: str = "cpu", tta: bool = False):
    """
    hd-bet 2.x with single-input:
      - `-o` MUST be a filename (not a directory).
      - mask file is created by appending `_mask` to the base of `-o`.
      - We pass a unique temp filename, then rename the produced mask to `out_mask`.
    """
    bin_path = _resolve_hdbet()
    _ensure_hdbet_weights()
    image = Path(image)
    out_mask = Path(out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    # unique temp base filename (hd-bet will append `_mask`)
    tag = uuid.uuid4().hex
    tmp_base = out_mask.parent / f"hdbet_{tag}.nii.gz"

    cmd = [
        bin_path,
        "-i", str(image),
        "-o", str(tmp_base),
        "-device", device,
        "--save_bet_mask",
        "--no_bet_image",
        "--verbose",
    ]
    if not tta:
        cmd.append("--disable_tta")

    # point HOME to where release_2.0.0 is copied
    env = dict(os.environ)
    env["HOME"] = os.environ.get("HDBET_HOME", "/opt/hd-bet-home")
    # belt & suspenders for offline
    env.update({"http_proxy":"", "https_proxy":"", "NO_PROXY":"*"})

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    # Show logs even on success
    logging.info("[hd-bet] cmd: %s", " ".join(cmd))
    if proc.stdout.strip():
        logging.info("[hd-bet] stdout:\n%s", proc.stdout.strip())
    if proc.stderr.strip():
        logging.info("[hd-bet] stderr:\n%s", proc.stderr.strip())

    if proc.returncode != 0:
        raise PreprocessError(
            f"hd-bet failed (rc={proc.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    # hd-bet will have written hdbet_<tag>_bet.nii.gz next to tmp_base
    expected_mask_path = out_mask.parent / f"hdbet_{tag}_bet.nii.gz"
    if not expected_mask_path.exists():
        files = [p.name for p in out_mask.parent.iterdir()]
        raise PreprocessError(
            f"hd-bet completed but no mask file found near {tmp_base}. "
            f"Directory listing: {files}"
        )

    # Take the first candidate and move/rename to the requested path
    if out_mask.exists() and out_mask.is_dir():
        shutil.rmtree(out_mask, ignore_errors=True)
    shutil.move(str(expected_mask_path), str(out_mask))

    # Clean up tmp base if hd-bet wrote any companion file with that exact name
    try:
        if tmp_base.exists():
            tmp_base.unlink()
    except Exception:
        pass

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
    if a.ndim == 5:  # (B,C,H,W,D)
        a = a[0]
    if a.ndim == 4:  # (C,H,W,D)
        a = a[0]
    # now (H,W,D)
    return (a >= threshold).astype(np.uint8)

def _collect_inv_transforms(reg_dir: Path) -> list[str]:
    items = sorted(reg_dir.glob("inv_*"), key=lambda p: int(p.stem.split("_")[1]))
    return [str(p) for p in items]

def preprocess_inputs(
    inputs: list[Path] | list[Path | None],
    mods: list[str],
    work_dir: Path| None = None,
    ref_mod: str = "flair",
    tmpl_path: Path = Path("templates/icbm_mni152_t1_09a_asym_bet.nii.gz"),
    do_bet: bool = True,
    do_reg: bool = True,
    ) -> None:
    """
    Preprocess input images for FOMO25 tasks.

    Assumes all modalities are co-registered and on the same grid. 

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
        FileNotFoundError: If template does not exist.
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

    # Check template
    if do_reg and not tmpl_path.exists():
        raise FileNotFoundError(f"Template not found: {tmpl_path}")
    
    # Map modalities to input paths
    mod2path: dict[str, Path] = {m.lower(): Path(p) for m, p in zip(mods, inputs)}

    ref_path = mod2path[ref_mod]
    ref_img = _load_nifti(ref_path)

    # Skull-stripping via HD-BET
    if do_bet:
        logging.info(f"Running hd-bet to get brain mask from {ref_path}...")
        mask_path = mask_dir / "brain_mask.nii.gz"
        use_gpu = torch.cuda.is_available()
        _run_hdbet_cli(ref_path, mask_path, device=("cuda" if use_gpu else "cpu"), tta=False)
        logging.info(f"hd-bet completed.")
        mask_img = _load_nifti(mask_path)

    # Registration to template
    if do_reg:
        logging.info(f"Computing registration fields of {ref_path} to template {tmpl_path}...")
        fixed = _load_nifti(tmpl_path)
        moving = _apply_mask(ref_img, mask_img) if do_bet else ref_img
        fwd, inv = _get_reg_transforms(moving, fixed)
        logging.info(f"Registration fields computed.")
        logging.info(f"FWD: {fwd}")
        logging.info(f"INV: {inv}")
        for i, t in enumerate(fwd):
            shutil.copy2(t, reg_dir / f"fwd_{i}{Path(t).suffix}")
        for i, t in enumerate(inv):
            shutil.copy2(t, reg_dir / f"inv_{i}{Path(t).suffix}")

    # Preprocess all inputs
    ref_spacing = ref_img.spacing
    for img_path in inputs:
        logging.info(f"Applying preprocessing to {img_path}...")
        try:
            img = _load_nifti(img_path)
            logging.info(f"Image loaded: {img}")
            if ref_spacing != img.spacing:
                raise ValueError(f"Mismatched spacing: {ref_spacing} != {img.spacing} "
                                 f"between modalities")
            if do_bet:
                img = _apply_mask(img, mask_img)
                logging.info(f"Mask applied: {img}")
            if do_reg:
                img = _apply_transforms(img, fixed, fwd)
                logging.info(f"Transform applied: {img}")
            _save_nifti(img, in_dir / img_path.name)
            logging.info(f"Preprocessing of {img_path} completed.")
        except Exception as e:
            raise PreprocessError(f"Failed to preprocess {img_path}: {e}")

def revert_preprocess(
    pred: torch.Tensor,
    orig: str | Path,
    work_dir: Path| None = None,
    do_reg: bool = True,
    tmpl_path: Path = Path("templates/icbm_mni152_t2_09a_asym_bet.nii.gz"),
) -> ants.ANTsImage:
    """
    Revert preprocessing on predicted segmentation masks for FOMO25 tasks.

    Assumes that following `predict_transforms` are already reversed:
    - Padding
    - Cropping
    - Resampling
    - Orientation (flipping)

    Steps: 
    1) Convert to uint8 numpy array
    2) Load fwd/inv transforms from reg.mat
    3) Apply transforms to pred

    Args:
        pred: Predicted segmentation mask as torch.Tensor.
        orig_img: Original image as str or Path.
        work_dir: Optional target directory for preprocessed images.

    Returns:
        Reversed segmentation mask as ants.ANTsImage.
    """
    work_dir = Path(work_dir or os.getenv("ANYBRAINER_CACHE", "/tmp/anyBrainer"))

    orig_path = Path(orig)
    if not orig_path.exists():
        raise FileNotFoundError(f"Original image not found: {orig_path}")

    orig_img = _load_nifti(orig_path)
    logging.info(f"Original image loaded: {orig_img}")
    pred_arr = _tensor_to_3d_mask_binary(pred)
    if do_reg:
        inv_transforms = _collect_inv_transforms(work_dir / "reg")
        if len(inv_transforms) == 0:
            raise FileNotFoundError("Inverse transforms not found.")
        
        if not tmpl_path.exists():
            raise FileNotFoundError(f"Template not found: {tmpl_path}")

        tmpl_img = _load_nifti(tmpl_path)
        logging.info(f"Template image loaded: {tmpl_img}")
        pred_img = _numpy_to_ants(pred_arr, tmpl_img)
        logging.info(f"Pred image loaded: {pred_img}")
        reverted = _apply_transforms(pred_img, orig_img, inv_transforms, interp='nearestNeighbor')
        logging.info(f"Reverted image: {reverted}")
        return reverted
    
    logging.info(f"No registration applied.")
    return _numpy_to_ants(pred_arr, orig_img)