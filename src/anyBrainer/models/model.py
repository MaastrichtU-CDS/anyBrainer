"""
PyTorch Lightning Modules to control all model-specific operations, 
such as training, validation, and testing steps. 

Includes:
- BaseModel: Base class for all models
- MAEModel: Model for masked autoencoder
- ContrastiveModel: Model for contrastive learning

The model is responsible for:
- Defining the model architecture
- Defining the loss function
- Defining the optimizer
- Defining the learning rate scheduler
- Defining the training, validation, and testing steps
"""

__all__ = [
    "BaseModel",
]

import logging
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import anyBrainer.models.networks as nets


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base class for all models."""

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict,
        *,
        init_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
        **kwargs,
    ):
        super().__init__()
        self.model = self._get_model(model_name, model_kwargs)
        
        if init_fn is not None:
            self.model.apply(init_fn)
            ignore_hparams.append("init_fn")
        
        self.save_hyperparameters(ignore=ignore_hparams, logger=len(self.loggers) > 0)
    
    def _get_model(self, model_name: str, model_kwargs: dict) -> nn.Module:
        """Get model from anyBrainer.models.networks."""
        try:
            model_cls = getattr(nets, model_name)
        except AttributeError:
            msg = f"Model '{model_name}' not found in networks module."
            logger.error(msg)
            raise ValueError(msg)
        
        if not issubclass(model_cls, nn.Module):
            msg = f"Retrieved object '{model_name}' is not a subclass of nn.Module."
            logger.error(msg)
            raise TypeError(msg)
        
        model = model_cls(**model_kwargs)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        raise NotImplementedError("Training step not implemented")
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        raise NotImplementedError("Validation step not implemented")
    
    def test_step(self, batch: dict, batch_idx: int):
        """Test step."""
        raise NotImplementedError("Test step not implemented")
    
    def predict_step(self, batch: dict, batch_idx: int):
        """Predict step."""
        raise NotImplementedError("Predict step not implemented")
    
    def configure_optimizers(self):
        """Configure optimizers."""
        raise NotImplementedError("Configure optimizers not implemented")
    
    def count_parameters(self, trainable: bool = False):
        """Count parameters."""
        if trainable:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())


class Swinv2CLModel(BaseModel):
    """Swinv2CL model."""

    def __init__(
        self,
        *,
        model_kwargs: dict = {},
        init_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
        **kwargs,
    ):
        super().__init__(
            "Swinv2CL",
            model_kwargs,
            init_fn=init_fn,
            ignore_hparams=ignore_hparams,
            **kwargs,
        )

    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        raise NotImplementedError("Training step not implemented")
