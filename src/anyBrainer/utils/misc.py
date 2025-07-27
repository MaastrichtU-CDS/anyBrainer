"""
Utility functions for the project.
"""

__all__ = [
    "resolve_path",
    "setup_seed",
    "find_checkpoint",
    "load_pretrained_weights",
]

import os
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from functools import partial

import torch
import lightning.pytorch as pl

logger = logging.getLogger(__name__)

def resolve_path(path: Path | str) -> Path:
    """Expand user and resolve path."""
    return Path(path).expanduser().resolve()


def setup_seed(continue_from_most_recent=False):
    """Set up a random seed for reproducibility."""
    if not continue_from_most_recent:
        dt = datetime.datetime.now()
        seed = int(dt.strftime("%m%d%H%M%S"))
    else:
        seed = None  # Will be loaded from checkpoint if available

    pl.seed_everything(seed=seed, workers=True)
    return torch.initial_seed()


def find_checkpoint(version_dir, continue_from_most_recent):
    """Find the latest checkpoint if continuing training."""
    checkpoint_path = None
    if continue_from_most_recent:
        potential_checkpoint = os.path.join(version_dir, "checkpoints", "last.ckpt")
        if os.path.isfile(potential_checkpoint):
            checkpoint_path = potential_checkpoint
            logging.info(
                "Using last checkpoint and continuing training: %s", checkpoint_path
            )
    return checkpoint_path


def load_pretrained_weights(weights_path, compile_flag):
    """Load pretrained weights with handling for compiled models and PyTorch Lightning checkpoints."""
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Extract the state_dict from PyTorch Lightning checkpoint if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading from PyTorch Lightning checkpoint")
        state_dict = checkpoint["state_dict"]
    else:
        print("Loading from standard model checkpoint")
        state_dict = checkpoint

    # Handle compiled checkpoints when loading to uncompiled model
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if "_orig_mod" in first_key and not compile_flag:
            print("Converting compiled model weights to uncompiled format")
            uncompiled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("_orig_mod.", "")
                uncompiled_state_dict[new_key] = state_dict[key]
            state_dict = uncompiled_state_dict

    return state_dict