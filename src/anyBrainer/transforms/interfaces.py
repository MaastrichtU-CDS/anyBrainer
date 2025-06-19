"""
Abstract interfaces for transform builders and managers.
"""

__all__ = [
    "TransformBuilderInterface",
    "TransformManagerInterface",
]
    
from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence
import logging

# pyright: reportPrivateImportUsage=false
from monai.transforms import Transform

logger = logging.getLogger(__name__)


class TransformBuilderInterface(ABC):
    """Abstract interface for transform builders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def build(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        """Build and return the transform."""
        pass


class TransformManagerInterface(ABC):
    """Abstract interface for transform managers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def get_train_transforms(self) -> Transform:
        """Get training transforms."""
        pass
    
    @abstractmethod
    def get_val_transforms(self) -> Transform:
        """Get validation transforms."""
        pass