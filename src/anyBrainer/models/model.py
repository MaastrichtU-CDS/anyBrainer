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
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

import anyBrainer.models.networks as nets
import anyBrainer.models.schedulers as schedulers
from anyBrainer.models.losses import InfoNCELoss
from anyBrainer.models.utils import (
    modality_to_onehot,
    top1_accuracy,
    get_inferer_from_roi_size,
)


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
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        other_schedulers: list[dict[str, Any]] = [],
        weights_init_fn: Callable | None = None,
        logits_postprocess_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
    ):
        super().__init__()
        self.model = self._get_model(model_name, model_kwargs)
        self.optimizer_kwargs = optimizer_kwargs # initialized in configure_optimizers
        self.lr_scheduler_kwargs = lr_scheduler_kwargs # initialized in configure_optimizers
        self.other_schedulers_step, self.other_schedulers_epoch = (
            self._get_other_schedulers(other_schedulers)
        )
        self.logits_postprocess_fn = logits_postprocess_fn
        
        if weights_init_fn is not None:
            self.model.apply(weights_init_fn)
            ignore_hparams.append("weights_init_fn")
        
        if self.logits_postprocess_fn is not None:
            ignore_hparams.append("logits_postprocess_fn")
        
        self.save_hyperparameters(ignore=["ignore_hparams"] + ignore_hparams, logger=True)
    
    def _get_model(self, model_name: str, model_kwargs: dict) -> nn.Module:
        """Get model from anyBrainer.models.networks."""
        # Extract requested model class
        try:
            model_cls = getattr(nets, model_name)
        except AttributeError:
            msg = f"Model '{model_name}' not found in networks module."
            logger.error(msg)
            raise ValueError(msg)
        
        # Ensure model is a subclass of nn.Module
        if not issubclass(model_cls, nn.Module):
            msg = f"Retrieved object '{model_name}' is not a subclass of nn.Module."
            logger.error(msg)
            raise TypeError(msg)
        
        # Handle improper initialization args
        try:
            model = model_cls(**model_kwargs)
        except Exception as e:
            msg = f"Error initializing model '{model_name}': {e}"
            logger.exception(msg)
            raise
        
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
        # Recursive calls to handle multiple optimizers
        if isinstance(optimizer_kwargs, list):
            optimizers_list = []
            for optimizer_kwargs in optimizer_kwargs:
                optimizers_list.append(self._get_optimizer(optimizer_kwargs, model))
            return optimizers_list
        
        # Ensure required keys are provided
        if "name" not in optimizer_kwargs:
            msg = "Optimizer name not found in optimizer_kwargs."
            logger.error(msg)
            raise ValueError(msg)

        optimizer_kwargs = optimizer_kwargs.copy()
        
        # Extract requested optimizer class
        try:
            cls_name = optimizer_kwargs.pop("name")
            optimizer_cls = getattr(optim, cls_name)
        except AttributeError:
            msg = f"Optimizer '{cls_name}' not found in torch.optim."
            logger.error(msg)
            raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        except Exception as e:
            msg = f"Error initializing optimizer '{cls_name}': {e}"
            logger.exception(msg)
            raise

        return optimizer
    
    def _get_lr_scheduler(
        self,
        lr_scheduler_kwargs: dict | list[dict],
        optimizer: optim.Optimizer | list[optim.Optimizer],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Get LR scheduler from torch.optim.lr_scheduler or anyBrainer.models.schedulers.

        The returned LR scheduler is a dictionary with the keys expected in Lightning's lr_scheduler_config.

        Special cases:
        - If multiple LR schedulers and optimizers are provided, return a list of lr_scheduler dictionaries.
        - If a single LR scheduler is provided with multiple optimizers, return a list of lr_scheduler
          dictionaries for each optimizer.

        Raises:
        - ValueError: If lr_scheduler_kwargs is a list and optimizer is not a list.
        - ValueError: If lr_scheduler_kwargs is a list and optimizer is a list but the lengths do not match.
        - ValueError: If lr_scheduler name is not found in lr_scheduler_kwargs.
        - ValueError: If scheduler name is not found in torch.optim.lr_scheduler or
                      anyBrainer.models.schedulers.
        - ValueError: If lr_scheduler name, interval, and frequency are not provided in lr_scheduler_kwargs.
        """
        # Recursive calls to handle multiple LR schedulers
        if isinstance(lr_scheduler_kwargs, list):
            if not isinstance(optimizer, list):
                msg = "Optimizer must be a list if lr_scheduler_kwargs is a list."
                logger.error(msg)
                raise ValueError(msg)
            
            if len(lr_scheduler_kwargs) != len(optimizer):
                msg = "Length of lr_scheduler_kwargs and optimizer must match."
                logger.error(msg)
                raise ValueError(msg)
            
            schedulers_list = []
            for _scheduler, _optimizer in zip(lr_scheduler_kwargs, optimizer):
                schedulers_list.append(self._get_lr_scheduler(_scheduler, _optimizer))
            return schedulers_list
        
        if not isinstance(lr_scheduler_kwargs, list) and isinstance(optimizer, list):
            schedulers_list = []
            for _optimizer in optimizer:
                schedulers_list.append(self._get_lr_scheduler(lr_scheduler_kwargs, _optimizer))
            return schedulers_list
        
        # Ensure required keys are provided
        if "name" not in lr_scheduler_kwargs:
            msg = "Scheduler name not found in lr_scheduler_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        required_keys = {"name", "interval", "frequency"}
        if required_keys.issubset(lr_scheduler_kwargs.keys()):
            lr_scheduler_kwargs = lr_scheduler_kwargs.copy()
            lr_scheduler_dict = {
                "name": lr_scheduler_kwargs.pop("name"),
                "interval": lr_scheduler_kwargs.pop("interval"),
                "frequency": lr_scheduler_kwargs.pop("frequency"),
                "monitor": lr_scheduler_kwargs.pop("monitor", None),
                "strict": lr_scheduler_kwargs.pop("strict", False),
            }
        else:
            msg = "LR scheduler 'name', 'interval', and 'frequency' must be provided."
            logger.error(msg)
            raise ValueError(msg)
        
        # Extract requested LR scheduler class
        try:
            lr_scheduler_cls = getattr(optim.lr_scheduler, lr_scheduler_dict["name"])
        except AttributeError:
            try:
                lr_scheduler_cls = getattr(schedulers, lr_scheduler_dict["name"])
            except AttributeError:  
                msg = (f"LR scheduler '{lr_scheduler_dict['name']}' not found in "
                       f"torch.optim.lr_scheduler or anyBrainer.models.schedulers.")
                logger.error(msg)
                raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)
            lr_scheduler_dict["scheduler"] = lr_scheduler
        except Exception as e:
            msg = f"Error initializing LR scheduler '{lr_scheduler_dict['name']}': {e}"
            logger.exception(msg)
            raise 

        return lr_scheduler_dict
    
    def _get_other_schedulers(
        self,
        other_schedulers: list[dict[str, Any]],
    ) -> tuple[list[schedulers.ParameterScheduler], list[schedulers.ParameterScheduler]]:
        """
        Get any other custom schedulers from anyBrainer.models.schedulers.
        
        Groups the schedulers into step and epoch schedulers, the values of which are
        extracted using built-in Lightning hooks.
        Always assuming list of dicts.

        Raises:
        - ValueError: If scheduler name is not found in anyBrainer.models.schedulers.
        """
        other_schedulers_step = []
        other_schedulers_epoch = []
        for scheduler_kwargs in other_schedulers:
            # Ensure required keys are provided
            if "name" not in scheduler_kwargs:
                msg = "Scheduler name not found in scheduler_kwargs."
                logger.error(msg)
                raise ValueError(msg)
            
            if "interval" not in scheduler_kwargs:
                msg = "Scheduler interval not found in scheduler_kwargs."
                logger.error(msg)
                raise ValueError(msg)
            
            scheduler_kwargs = scheduler_kwargs.copy()
            
            # Extract requested scheduler class
            try:
                scheduler_name = scheduler_kwargs.pop("name")
                scheduler_interval = scheduler_kwargs.pop("interval")
                scheduler_cls = getattr(schedulers, scheduler_name)
            except AttributeError:
                msg = f"Scheduler '{scheduler_name}' not found in anyBrainer.models.schedulers."
                logger.error(msg)
                raise ValueError(msg)   
            
            # Handle improper initialization args
            try:
                scheduler = scheduler_cls(**scheduler_kwargs)
            except Exception as e:
                msg = f"Error initializing scheduler '{scheduler_name}': {e}"
                logger.exception(msg)
                raise
            
            # Add scheduler to list
            if scheduler_interval == "step":
                other_schedulers_step.append(scheduler)
            elif scheduler_interval == "epoch":
                other_schedulers_epoch.append(scheduler)
            else:
                msg = f"Scheduler interval '{scheduler_interval}' not supported."
                logger.error(msg)
                raise ValueError(msg)
        
        return other_schedulers_step, other_schedulers_epoch
    
    def configure_optimizers(
        self,
    ) -> (optim.Optimizer | dict[str, Any] | list[optim.Optimizer] | 
          tuple[list[optim.Optimizer], list[dict[str, Any]]]):
        """
        Configure optimizers.
        
        Aligning with Lighnting's documentation, it can return the following:
        (depending on the optimizer_kwargs and scheduler_kwargs inputs)
        - A single optimizer
        - A dict with a single optimizer and a single lr_scheduler_config
        - A list of optimizers
        - A list of optimizers with a list of lr_scheduler_configs
        """
        optimizer = self._get_optimizer(self.optimizer_kwargs, self.model)

        if self.lr_scheduler_kwargs is not None:
            lr_scheduler = self._get_lr_scheduler(self.lr_scheduler_kwargs, optimizer)
        else:
            lr_scheduler = None

        if isinstance(optimizer, list) and lr_scheduler is not None:
            return optimizer, lr_scheduler # type: ignore
        
        if isinstance(optimizer, list) and lr_scheduler is None:
            return optimizer
        
        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }
        
        return optimizer
    
    def get_step_scheduler_values(self) -> list[dict[str, Any]]:
        """
        Get values for step-based schedulers.
        
        Assumes custom schedulers contain a get_value method that accepts 
        the current global step as an argument.
        """
        if not hasattr(self, "global_step"):
            msg = "Global step not found in model."
            logger.error(msg)
            raise ValueError(msg)
        
        step_scheduler_values = []
        for scheduler in self.other_schedulers_step:
            step_scheduler_values.append(scheduler.get_value(self.global_step))
        return step_scheduler_values
    
    def get_epoch_scheduler_values(self) -> list[float]:
        """
        Get values for epoch-based schedulers.
        
        Assumes custom schedulers contain a get_value method that accepts 
        the current global epoch as an argument.
        """
        if not hasattr(self, "current_epoch"):
            msg = "Current epoch not found in model."
            logger.error(msg)
            raise ValueError(msg)
        
        epoch_scheduler_values = []
        for scheduler in self.other_schedulers_epoch:
            epoch_scheduler_values.append(scheduler.get_value(self.current_epoch))
        return epoch_scheduler_values

    def _count_params(self, trainable: bool = False):
        """Count parameters."""
        if trainable:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())
    
    def summarize_model(self):
        """Show model parameters."""
        out_msg = "#Model parameters#\n"
        for name, param in self.model.named_parameters():
            out_msg += f"{name:60s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}\n"
        
        out_msg += f"\n#Model buffers#\n"
        for name, b in self.model.named_buffers():
            out_msg += f"{name:55s} {tuple(b.shape)}\n"
        
        out_msg += f"\n#Model summary#\n"
        out_msg += f"Total parameters: {self._count_params()}\n"
        out_msg += f"Trainable parameters: {self._count_params(trainable=True)}\n"
        out_msg += f"Non-trainable parameters: {self._count_params(trainable=False)}\n"
                
        logger.info(out_msg)

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


class Swinv2CLModel(BaseModel):
    """Swinv2CL model."""
    def __init__(
        self,
        *,
        model_kwargs: dict,
        optimizer_kwargs: dict | list[dict],
        total_steps: int,
        lr_scheduler_kwargs: dict | list[dict] | None = None,
        weights_init_fn: Callable | None = None,
        logits_postprocess_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
        loss_kwargs: dict = {},
        loss_scheduler_kwargs: dict = {},
        momentum_scheduler_kwargs: dict = {},
        **kwargs,
    ):  
        if not isinstance(total_steps, int) or total_steps <= 0:
            msg = "Total steps must be a positive integer."
            logger.error(msg)
            raise ValueError(msg)
        
        other_schedulers = [
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "loss_weight",
                "start_step": int(total_steps * 
                                  loss_scheduler_kwargs.get("loss_weight_step_start_ratio", 0.05)),
                "end_step": int(total_steps *
                                loss_scheduler_kwargs.get("loss_weight_step_end_ratio", 0.1)),
                "start_value": loss_scheduler_kwargs.get("loss_weight_start_value", 0.0),
                "end_value": loss_scheduler_kwargs.get("loss_weight_end_value", 0.7),
                "mode": "linear",
            },
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "momentum",
                "start_step": 0,
                "end_step": int(total_steps *
                                momentum_scheduler_kwargs.get("momentum_step_end_ratio", 0.1)),
                "start_value": momentum_scheduler_kwargs.get("momentum_start_value", 0.99),
                "end_value": momentum_scheduler_kwargs.get("momentum_end_value", 0.999),
                "mode": "linear",
            },
        ]

        super().__init__(
            model_name="Swinv2CL",
            model_kwargs=model_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            other_schedulers=other_schedulers,
            logits_postprocess_fn=logits_postprocess_fn,
            weights_init_fn=weights_init_fn,
            ignore_hparams=ignore_hparams,
        )

        self.key_encoder = deepcopy(self.model)
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        
        # Initialize InfoNCE loss
        self.info_nce = InfoNCELoss(
            temperature=loss_kwargs.get("temperature", 0.1),
            top_k_negatives=loss_kwargs.get("top_k_negatives", None),
            postprocess_fn=logits_postprocess_fn,
        )

        # Initialize cross entropy loss
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=loss_kwargs.get("cross_entropy_weight", None),
        )
        
        # Initialize queue
        self.queue_size = loss_kwargs.get("queue_size", 16384)
        self.model.register_buffer("queue", torch.empty(0))

        logger.info(f"\nLightning module initialized with following "
                    f"hyperparameters:\n{self.hparams}")

    def _update_queue(self, q: torch.Tensor) -> None:
        """Update queue."""
        if self.queue.numel() == 0:
            self.queue = q.detach()
        else:
            self.queue = torch.cat([self.queue, q.detach()], dim=0)
            self.queue = self.queue[-self.queue_size:]
        
        self.log("train/queue_size", self.queue.shape[0], on_step=True, prog_bar=False, 
                 sync_dist=self.trainer.world_size > 1)
    
    @torch.no_grad()
    def _update_key_encoder(self):
        """Update key encoder."""
        momentum = self.get_step_scheduler_values()[1]["momentum"]

        params_q = list(self.model.parameters())
        params_k = list(self.key_encoder.parameters())
        
        if len(params_q) != len(params_k):
            msg = "Query and key encoder param counts differ."
            logger.error(msg)
            raise RuntimeError(msg)
        
        for param_q, param_k in zip(params_q, params_k):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)
        
        self.log("train/momentum", momentum, on_step=True, prog_bar=False, 
                 sync_dist=self.trainer.world_size > 1)

    def _compute_loss(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        q_aux: torch.Tensor,
        aux_spr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined InfoNCE loss with auxiliary CE loss."""
        loss_info_nce = self.info_nce(q_proj, k_proj)
        loss_aux = self.cross_entropy(q_aux, aux_spr)

        loss_weight = self.get_step_scheduler_values()[0]["loss_weight"]
        
        combined_loss = (loss_info_nce * loss_weight + 
                         loss_aux * (1 - loss_weight))
        
        return combined_loss, {"loss_info_nce": loss_info_nce, 
                               "loss_aux": loss_aux,
                               "loss_weight": loss_weight}

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        """Get modality one-hot labels to device."""
        if dataloader_idx != 3: # not for prediction
            batch["aux_labels"] = modality_to_onehot(batch, "mod", batch["query"].device)
        return batch

    def on_before_optimizer_step(self, optimizer: optim.Optimizer, optimizer_idx: int) -> None:
        """Log all current learning rates."""
        for i, _optimizer in enumerate(self.trainer.optimizers):
            for j, group in enumerate(_optimizer.param_groups):
                self.log(f"train/lr/opt{i}_group{j}", group["lr"], on_step=True, prog_bar=False, 
                         sync_dist=self.trainer.world_size > 1)
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._update_key_encoder()
    
    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        loss, loss_dict = self._compute_loss(q_proj, k_proj, q_aux, batch["aux_labels"])
        self.log_dict({
            "train/loss": loss,
            "train/loss_info_nce": loss_dict["loss_info_nce"],
            "train/loss_aux": loss_dict["loss_aux"],
            "train/loss_weight": loss_dict["loss_weight"],
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.trainer.world_size > 1)

        self._update_queue(q_proj)
            
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        loss, loss_dict = self._compute_loss(q_proj, k_proj, q_aux, batch["aux_labels"])
        acc = top1_accuracy(q_proj, batch["aux_labels"])
        
        self.log_dict({
            "val/loss": loss,
            "val/loss_info_nce": loss_dict["loss_info_nce"],
            "val/loss_aux": loss_dict["loss_aux"],
            "val/loss_weight": loss_dict["loss_weight"],
            "val/aux_acc": acc,
        }, on_epoch=True, prog_bar=True, sync_dist=self.trainer.world_size > 1)
    
    def test_step(self, batch: dict, batch_idx: int):
        """Test step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        loss, loss_dict = self._compute_loss(q_proj, k_proj, q_aux, batch["aux_labels"])
        acc = top1_accuracy(q_proj, batch["aux_labels"])
        
        self.log_dict({
            "test/loss": loss,
            "test/loss_info_nce": loss_dict["loss_info_nce"],
            "test/loss_aux": loss_dict["loss_aux"],
            "test/loss_weight": loss_dict["loss_weight"],
            "test/aux_acc": acc,
        }, on_epoch=True, prog_bar=True, sync_dist=self.trainer.world_size > 1)

