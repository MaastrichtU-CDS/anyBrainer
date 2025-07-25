"""Utility functions for I/O operations."""

__all__ = [
    "resolve_path",
    "create_save_dirs",
    "get_ckpt_path",
    "load_model_from_ckpt",
]

import logging
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def resolve_path(path: Path | str) -> Path:
    """Expand user and resolve path."""
    return Path(path).expanduser().resolve()

def create_save_dirs(
    exp_name: str, 
    root_dir: str | Path,
    new_version: bool,
    create_ckpt_dir: bool,
) -> None:
    """
    Create all required saving directories for an experiment.

    Args:
        exp_name: Name of the experiment.
        root_dir: Root directory for saving.
        new_version: Whether to create a new version of the experiment.
        create_ckpt_dir: Whether to create a checkpoint directory.
    """
    root_dir = resolve_path(root_dir)
    _create_dir(root_dir, new_version)
    
    exp_dir = root_dir / exp_name
    _create_dir(exp_dir, new_version)

    logs_dir = root_dir / "logs"
    _create_dir(logs_dir, new_version)
    
    if create_ckpt_dir:
        ckpt_dir = exp_dir / "checkpoints"
        _create_dir(ckpt_dir, new_version)

def _create_dir(path: Path, new_version: bool) -> None:
    """Create a directory if it does not exist."""
    try:
        path.mkdir(parents=True, exist_ok=not new_version)
    except FileExistsError:
        msg = (f"Directory {path} already exists; "
               "delete it to create a new version.")
        logger.error(msg)
        raise

def get_ckpt_path(
    save_dir: Path,
    model_checkpoint: Path | None = None,
) -> Path:
    """Get the path to the checkpoint file."""
    if model_checkpoint is None:
        return save_dir / "checkpoints" / "last.ckpt"
    return model_checkpoint

def load_model_from_ckpt(
    model_cls: type[pl.LightningModule],
    ckpt_path: Path,
) -> pl.LightningModule | None:
    """Load a model from a checkpoint file."""
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint file {ckpt_path} does not exist; "
                       "will create new model.")
        return
    
    try:
        return model_cls.load_from_checkpoint(ckpt_path)
    except Exception:
        logger.exception(f"Failed to load model from checkpoint; "
                         "will create new model.")
        return