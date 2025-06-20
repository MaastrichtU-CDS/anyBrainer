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
        self._params = {}
    
    @property
    def params(self) -> Dict[str, Any]:
        """Get the parameters used in the last build() call."""
        return self._params.copy()  # Return a copy to prevent external modification
    
    def update_params(self, name: str, new_params: Dict[str, Any]):
        """Update the parameters used in the last build() call."""
        self._params[name] = new_params
    
    def build(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        """Build and return the transform."""
        # Reset params at the start of each build
        self._params = {}

        return self._build_transforms(img_keys, allow_missing_keys)
    
    @abstractmethod
    def _build_transforms(self, img_keys: Sequence[str], allow_missing_keys: bool = False) -> Transform:
        """Build and return the transform. Implement this in concrete classes."""
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