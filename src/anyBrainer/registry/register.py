"""Registry for anyBrainer core components."""

__all__ = [
    "RegistryKind",
    "register",
    "get",
    "flush",
]

import logging
from enum import Enum
from copy import deepcopy
from typing import Callable, TypeVar, Any
from collections.abc import Callable as ABCCallable

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.optim.lr_scheduler import _LRScheduler
from monai.inferers.inferer import Inferer

from anyBrainer.registry.utils import check_allowed_types
from anyBrainer.interfaces import (
    DataExplorer,
    LoggingManager,
    ParameterScheduler,
    Workflow,
    PLModuleMixin
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

class RegistryKind(str, Enum):
    """Kind of registry."""
    WORKFLOW        = "workflow"
    DATA_EXPLORER   = "data_explorer"
    LOGGING_MANAGER = "logging_manager"
    TRAINER         = "trainer"
    PL_MODULE       = "pl_module"
    PL_MODULE_MIXIN = "pl_module_mixin"
    DATAMODULE      = "datamodule"
    NETWORK         = "network"
    LOSS            = "loss"
    OPTIMIZER       = "optimizer"
    LR_SCHEDULER    = "lr_scheduler"
    PARAM_SCHEDULER = "param_scheduler"
    INFERER         = "inferer"
    TRANSFORM       = "transform"
    CALLBACK        = "callback"
    METRIC          = "metric"
    UTIL            = "util"


REGISTRIES: dict[RegistryKind, dict[str, object]] = {
    kind: {} for kind in RegistryKind
}

ALLOWED_TYPES: dict[RegistryKind, tuple[object, ...]] = {
    RegistryKind.DATA_EXPLORER:   (DataExplorer,),
    RegistryKind.LOGGING_MANAGER: (LoggingManager,),
    RegistryKind.PARAM_SCHEDULER: (ParameterScheduler,),
    RegistryKind.WORKFLOW:        (Workflow,),
    RegistryKind.METRIC:          (type,),
    RegistryKind.TRAINER:         (pl.Trainer,),
    RegistryKind.PL_MODULE:       (pl.LightningModule,),
    RegistryKind.PL_MODULE_MIXIN: (PLModuleMixin,),
    RegistryKind.DATAMODULE:      (pl.LightningDataModule,),
    RegistryKind.NETWORK:         (nn.Module,),
    RegistryKind.LOSS:            (nn.Module,),
    RegistryKind.OPTIMIZER:       (optim.Optimizer,),
    RegistryKind.LR_SCHEDULER:    (_LRScheduler,),
    RegistryKind.CALLBACK:        (pl.Callback,),
    RegistryKind.INFERER:         (Inferer,),
    RegistryKind.TRANSFORM:       (ABCCallable,),
    RegistryKind.UTIL:            (ABCCallable,),
}

# Register interface
def register(kind: RegistryKind) -> Callable[[T], T]:
    """Decorator that records the object and returns it unchanged."""
    def _decorator(obj: T) -> T:
        check_allowed_types(kind.value, obj, ALLOWED_TYPES[kind])
        bucket = REGISTRIES[kind]
        name = getattr(obj, "__name__", repr(obj))
        if name in bucket:
            msg = f"{name} already registered in {kind}"
            logger.error(msg)
            raise ValueError(msg)
        bucket[name] = obj
        return obj
    return _decorator

def get(kind: RegistryKind, name: str) -> Any:
    """Get an object from the registry."""
    try:
        return REGISTRIES[kind][name]
    except KeyError:
        msg = f"{name!r} not found in {kind.value} registry"
        logger.error(msg)
        raise ValueError(msg)

def flush(
    kind: RegistryKind | None = None
) -> dict[str, Any] | dict[RegistryKind, dict[str, Any]]:
    """Return a copy of the registry contents."""
    if kind is not None:
        if kind not in REGISTRIES:
            raise KeyError(f"Unknown registry kind: {kind}")
        return deepcopy(REGISTRIES[kind])
    return deepcopy(REGISTRIES)