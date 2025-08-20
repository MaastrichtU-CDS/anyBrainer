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
from typing import Any, Sequence

import torch
import torch.nn as nn
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
    model_instance: nn.Module,
    checkpoint_path: Path,
    select_prefixes: str | Sequence[str] | None = None,
    rename_map: dict[str, str] | None = None,
    strict: bool = False,
    torch_load_kwargs: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load (optionally a subset of) parameters from a checkpoint into a module,
    with optional prefix-based key renaming.

    Args:
        model_instance: The module to load parameters into.
        checkpoint_path: The path to the checkpoint file.
        select_prefixes: A list of prefixes to select parameters from.
        rename_map: A dictionary of old prefixes to new prefixes. 
        strict: Whether to raise an error if there are missing or unexpected keys.
        torch_load_kwargs: Additional keyword arguments to pass to `torch.load`.

    Returns:
        model_instance: The loaded module.
        stats: A dictionary of statistics.
    
    Raises:
        FileNotFoundError: if ``checkpoint_path`` does not exist.
        TypeError: if the loaded checkpoint does not contain a dict-like state dict.
    """
    kw = torch_load_kwargs or {}
    ckpt = torch.load(str(checkpoint_path), **kw)
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        msg = "No dict-like 'state_dict' in checkpoint."
        logger.error(msg)
        raise TypeError(msg)

    # selection
    if select_prefixes:
        if isinstance(select_prefixes, str):
            select_prefixes = [select_prefixes]
        selected = {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in select_prefixes)}
        ignored = [k for k in state_dict if k not in selected]
    else:
        selected = dict(state_dict)
        ignored = []

    # renaming
    to_load = {}
    if rename_map:
        for k, v in selected.items():
            new_k = k
            for old, new in rename_map.items():
                if k.startswith(old):
                    new_k = new + k[len(old):]
                    break
            to_load[new_k] = v
    else:
        to_load = selected

    result = model_instance.load_state_dict(to_load, strict=strict)
    if hasattr(result, "missing_keys"):
        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)
    elif isinstance(result, tuple) and len(result) == 2:
        missing_keys, unexpected_keys = list(result[0]), list(result[1])
    else:
        missing_keys, unexpected_keys = [], []

    stats = {
        "loaded_keys": list(to_load.keys()),
        "ignored_keys": ignored,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    return model_instance, stats