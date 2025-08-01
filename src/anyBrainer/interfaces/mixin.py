"""Interfaces for engine mixin classes."""

__all__ = [
    "PLModuleMixin",
]

from abc import ABC, abstractmethod

class PLModuleMixin(ABC):
    """Interface for Pytorch Lightning module mixin classes."""
    @abstractmethod
    def setup_mixin(self):
        """Initialize mixin; to be called in __init__ of the module."""
        pass