"""Utility functions for transforms"""

__all__ = [
    "DeterministicCompose",
    "reseed",
]

import logging

import numpy as np
import torch
from collections import deque
# pyright: reportPrivateImportUsage=false
from monai.transforms import Compose, Randomizable

logger = logging.getLogger(__name__)

def reseed(
    comp: Compose, 
    master_seed: int | None = None, 
    rng: np.random.RandomState | None = None,
) -> None:
    """
    Walk through `comp` (including any nested Compose) and give every
    Randomizable transform a unique, repeatable seed that is *derived*
    from `master_seed`.  Order of **construction** no longer matters.
    """
    # Save the *process-wide* RNG states so our reseeding doesn’t pollute them.
    _np_state    = np.random.get_state()
    _torch_state = torch.random.get_rng_state()
    
    rng = np.random.RandomState(master_seed) if rng is None else rng
    for t in comp.flatten().transforms:          # depth-first, left-to-right
        if isinstance(t, Randomizable):
            # Give every Randomizable an independent, reproducible seed.
            t.set_random_state(seed=int(rng.randint(2**32 - 1)))

    # Restore global RNGs so later object construction (or user code) is
    # unaffected by the per-transform reseeding above.
    np.random.set_state(_np_state)
    torch.random.set_rng_state(_torch_state)

class DeterministicCompose(Compose):
    def __init__(self, transforms, master_seed, **kwargs):
        super().__init__(transforms, **kwargs)
        reseed(self, master_seed)

