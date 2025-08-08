"""Utility functions for I/O operations."""

__all__ = [
    "resolve_path",
    "create_save_dirs",
    "load_model_from_ckpt",
    "load_config",
    "load_param_group_from_ckpt",
]

import logging
from pathlib import Path
from typing import Any

import torch
import lightning.pytorch as pl
import yaml
import json

logger = logging.getLogger(__name__)


def resolve_path(path: Path | str) -> Path:
    """Expand user and resolve path."""
    return Path(path).expanduser().resolve()

def create_save_dirs(
    exp_dir: Path,
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
    _create_dir(exp_dir, new_version=False)

    logs_dir = exp_dir / "logs"
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

def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON file into a Python dict."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed – cannot read YAML files.")
        return yaml.safe_load(path.read_text())  # type: ignore[arg-type]
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    raise RuntimeError(
        "Unsupported config format – use .yaml, .yml or .json",
    )

def load_param_group_from_ckpt(
    model_instance: torch.nn.Module,
    checkpoint_path: Path,
    param_group_prefix: str | list[str] | None = None,
    extra_load_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Selectively load a parameter group from a checkpoint file.
    """
    if extra_load_kwargs is None:
        extra_load_kwargs = {}

    # Get checkpoint
    ckpt = torch.load(checkpoint_path, **extra_load_kwargs)
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Determine which keys to load
    if param_group_prefix is None:
        filtered_dict = state_dict
    else:
        if isinstance(param_group_prefix, str):
            param_group_prefix = [param_group_prefix]
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in param_group_prefix)
        }

    ignored_keys = [k for k in state_dict.keys() if k not in filtered_dict]

    # Remove `model.` from target keys (directly loading to model not to pl module)
    filtered_dict = {
        k.split("model.")[-1]: v for k, v in filtered_dict.items()
    }

    missing_keys, unexpected_keys = model_instance.load_state_dict(
        filtered_dict, strict=extra_load_kwargs.get("strict", False)
    )

    stats = {
        "loaded_keys": list(filtered_dict.keys()),
        "ignored_keys": ignored_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }

    return model_instance, stats