"""Utility functions for parallel processing."""

__all__ = [
    "make_worker_init_fn",
]

import logging
import random
from typing import Any, Callable, Literal

import numpy as np
import torch
from monai.data.utils import set_rnd

logger = logging.getLogger(__name__)

def make_worker_init_fn(
    seed: int | None = None,
    setup_logging_fn: Callable[[], None] | None = None,
    seeding_fn: Callable[[Any, int], Any] | None = set_rnd,
    loader: Literal["train", "val", "test", "predict"] = "train",
) -> Callable[[int], None]:
    """Make a worker init function."""
    
    def custom_worker_init_fn(worker_id: int) -> None:
        """
        Initialize worker with logging setup and input file.
        """
        worker_info = torch.utils.data.get_worker_info()

        if setup_logging_fn:
            setup_logging_fn()

        # Compute base seed always
        if worker_info is not None:
            base_seed = seed + worker_info.id if seed is not None else worker_info.seed
        else:
            base_seed = seed if seed is not None else 0

        # Optional user seeding logic
        if seeding_fn and worker_info is not None:
            seeding_fn(worker_info.dataset, base_seed)

        # Always seed the standard RNGs
        np.random.seed(base_seed)
        random.seed(base_seed)
        torch.manual_seed(base_seed)

        logger.info(f"Worker {worker_id} initialized with seed {base_seed}", 
            extra={"wandb": {
                "_wandb_mode": "sync",
                f"{loader}/worker_seed/{worker_id}": base_seed,
            }}
        )
    
    return custom_worker_init_fn

