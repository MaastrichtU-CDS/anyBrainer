"""Registry for anyBrainer core components."""
from typing import Callable, TypeVar
from enum import Enum

class RegistryKind(str, Enum):
    """Kind of registry."""
    WORKFLOW        = "workflow"
    DATA_MANAGER    = "data_manager"
    LOGGING_MANAGER = "logging_manager"
    TRAINER         = "trainer"
    PL_MODULE       = "pl_module"
    DATAMODULE      = "datamodule"
    NETWORK         = "network"
    LOSS            = "loss"
    OPTIMIZER       = "optimizer"
    LR_SCHEDULER    = "lr_scheduler"
    PARAM_SCHEDULER = "param_scheduler"
    TRANSFORM       = "transform"
    CALLBACK        = "callback"
    METRIC          = "metric"
    UTIL            = "util"


_REGISTRIES: dict[RegistryKind, dict[str, object]] = {
    kind: {} for kind in RegistryKind
}

T = TypeVar("T")

def register(kind: RegistryKind) -> Callable[[T], T]:
    """Decorator that records the object and returns it unchanged."""
    def _decorator(obj: T) -> T:
        bucket = _REGISTRIES[kind]
        name = getattr(obj, "__name__", repr(obj))
        if name in bucket:
            raise ValueError(f"{name} already registered in {kind}")
        bucket[name] = obj
        return obj
    return _decorator

def get(kind: RegistryKind, name: str) -> object:
    try:
        return _REGISTRIES[kind][name]
    except KeyError:
        raise ValueError(f"{name!r} not found in {kind.value} registry")