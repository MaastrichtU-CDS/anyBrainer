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
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import anyBrainer.models.networks as nets
import anyBrainer.models.schedulers as schedulers


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """
    Base class for all models.
    
    This class is responsible for:
    - Defining the model architecture
    - Defining the optimizer
    - Defining the learning rate scheduler

    Args:
        model_name: Name of the model to use.
        model_kwargs: Keyword arguments for the model.
        optimizer_kwargs: Keyword arguments for the optimizer, including the optimizer name.
        scheduler_kwargs: Keyword arguments for the learning rate scheduler, including:
            - name: Name of the learning rate scheduler.
            - interval: Interval of the learning rate scheduler.
            - frequency: Frequency of the learning rate scheduler.
            - monitor (optional): Metric to monitor.
            - strict (optional): Whether to strict the learning rate scheduler.
            - other arguments: Other arguments for the learning rate scheduler.
            If None, no scheduler is used.
        weights_init_fn: Function to initialize the model weights.
        ignore_hparams: List of hyperparameters to ignore.
    
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict,
        optimizer_kwargs: dict | list[dict],
        scheduler_kwargs: dict | list[dict] | None = None,
        weights_init_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
        **kwargs,
    ):
        super().__init__()
        self.model = self._get_model(model_name, model_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        if weights_init_fn is not None:
            self.model.apply(weights_init_fn)
            ignore_hparams.append("weights_init_fn")
        
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
    
    def _get_optimizer(
        self,
        optimizer_kwargs: dict | list[dict],
        model: nn.Module,
    ) -> optim.Optimizer | list[optim.Optimizer]:
        """
        Get optimizer from torch.optim.

        If optimizer_kwargs is a list, return a list of optimizers through recursive calls.

        Raises:
        - ValueError: If optimizer name is not found in optimizer_kwargs.
        - ValueError: If optimizer name is not found in torch.optim.
        """
        if isinstance(optimizer_kwargs, list):
            optimizers_list = []
            for optimizer_kwargs in optimizer_kwargs:
                optimizers_list.append(self._get_optimizer(optimizer_kwargs, model))
            return optimizers_list
        
        if "name" not in optimizer_kwargs:
            msg = "Optimizer name not found in optimizer_kwargs."
            logger.error(msg)
            raise ValueError(msg)

        optimizer_kwargs = optimizer_kwargs.copy()
        
        try:
            cls_name = optimizer_kwargs.pop("name")
            optimizer_cls = getattr(optim, cls_name)
        except AttributeError:
            msg = f"Optimizer '{cls_name}' not found in torch.optim."
            logger.error(msg)
            raise ValueError(msg)
        
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        return optimizer
    
    def _get_scheduler(
        self,
        scheduler_kwargs: dict | list[dict],
        optimizer: optim.Optimizer | list[optim.Optimizer],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Get scheduler from torch.optim.lr_scheduler or anyBrainer.models.schedulers.

        The returned scheduler is a dictionary with the keys expected in Lightning's lr_scheduler_config.

        Special cases:
        - If multiple schedulers and optimizers are provided, return a list of scheduler dictionaries.
        - If a single scheduler is provided with multiple optimizers, return a list of scheduler
          dictionaries for each optimizer.

        Raises:
        - ValueError: If scheduler_kwargs is a list and optimizer is not a list.
        - ValueError: If scheduler_kwargs is a list and optimizer is a list but the lengths do not match.
        - ValueError: If scheduler name is not found in scheduler_kwargs.
        - ValueError: If scheduler name is not found in torch.optim.lr_scheduler or
                      anyBrainer.models.schedulers.
        - ValueError: If scheduler name, interval, and frequency are not provided in scheduler_kwargs.
        """
        if isinstance(scheduler_kwargs, list):
            if not isinstance(optimizer, list):
                msg = "Optimizer must be a list if scheduler_kwargs is a list."
                logger.error(msg)
                raise ValueError(msg)
            
            if len(scheduler_kwargs) != len(optimizer):
                msg = "Length of scheduler_kwargs and optimizer must match."
                logger.error(msg)
                raise ValueError(msg)
            
            schedulers_list = []
            for _scheduler, _optimizer in zip(scheduler_kwargs, optimizer):
                schedulers_list.append(self._get_scheduler(_scheduler, _optimizer))
            return schedulers_list
        
        if not isinstance(scheduler_kwargs, list) and isinstance(optimizer, list):
            schedulers_list = []
            for _optimizer in optimizer:
                schedulers_list.append(self._get_scheduler(scheduler_kwargs, _optimizer))
            return schedulers_list
        
        if "name" not in scheduler_kwargs:
            msg = "Scheduler name not found in scheduler_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        required_keys = {"name", "interval", "frequency"}
        if required_keys.issubset(scheduler_kwargs.keys()):
            scheduler_kwargs = scheduler_kwargs.copy()
            scheduler_dict = {
                "name": scheduler_kwargs.pop("name"),
                "interval": scheduler_kwargs.pop("interval"),
                "frequency": scheduler_kwargs.pop("frequency"),
                "monitor": scheduler_kwargs.pop("monitor", None),
                "strict": scheduler_kwargs.pop("strict", False),
            }
        else:
            msg = "Scheduler 'name', 'interval', and 'frequency' must be provided."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            scheduler_cls = getattr(optim.lr_scheduler, scheduler_dict["name"])
        except AttributeError:
            try:
                scheduler_cls = getattr(schedulers, scheduler_dict["name"])
            except AttributeError:  
                msg = (f"Scheduler '{scheduler_dict['name']}' not found in "
                       f"torch.optim.lr_scheduler or anyBrainer.models.schedulers.")
                logger.error(msg)
                raise ValueError(msg)
        
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        scheduler_dict["scheduler"] = scheduler

        return scheduler_dict
    
    def configure_optimizers(
        self,
    ) -> (optim.Optimizer | dict[str, Any] | list[optim.Optimizer] | 
          tuple[list[optim.Optimizer], list[dict[str, Any]]]):
        """
        Configure optimizers.
        
        Aligning with Lighnting's documentation, it can return the following:
        (depending on the optimizer_kwargs and scheduler_kwargs inputs)
        - A single optimizer
        - A dict with a single optimizer with a single lr_scheduler_config
        - A list of optimizers
        - A list of optimizers with a list of lr_scheduler_configs
        """
        optimizer = self._get_optimizer(self.optimizer_kwargs, self.model)

        if self.scheduler_kwargs is not None:
            scheduler = self._get_scheduler(self.scheduler_kwargs, optimizer)
        else:
            scheduler = None

        if isinstance(optimizer, list) and scheduler is not None:
            return optimizer, scheduler # type: ignore
        
        if isinstance(optimizer, list) and scheduler is None:
            return optimizer
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        
        return optimizer

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
    
    def count_params(self, trainable: bool = False):
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
        model_kwargs: dict,
        optimizer_kwargs: dict | list[dict],
        scheduler_kwargs: dict | list[dict] | None = None,
        weights_init_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
        **kwargs,
    ):
        super().__init__(
            model_name="Swinv2CL",
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            weights_init_fn=weights_init_fn,
            ignore_hparams=ignore_hparams,
            **kwargs,
        )

    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        raise NotImplementedError("Training step not implemented")
    