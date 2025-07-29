"""Logging manager interface."""

from __future__ import annotations

__all__ = [
    "LoggingManager",
]

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.loggers import WandbLogger

class LoggingManager(ABC):
    """Logging manager interface."""
    main_logger: logging.Logger
    wandb_logger: WandbLogger | None

    @abstractmethod
    def __init__(self, **settings: Any) -> None:
        """Initialize the logging manager with custom settings."""
        pass

    @abstractmethod
    def setup_main_logging(self) -> None:
        """Setup main logging."""
        pass

    @abstractmethod
    def get_setup_worker_logging_fn(self) -> Callable[[int], None]:
        """Get the function to setup worker logging."""

    @abstractmethod
    def close(self) -> None:
        """Close the logging manager."""
        pass