"""
Base PyTorch Lightning Module to control all model-specific operations, 
such as training, validation, and testing steps. 

The model is responsible for:
- Defining the model architecture
- Defining the loss function
- Defining the optimizer
- Defining the learning rate scheduler
- Defining the training, validation, and testing steps
"""

__all__ = [
    "BaseModel",
    "CLwAuxModel",
]

import logging
from typing import Any, Callable, cast
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl

from anyBrainer.engines.utils import (
    sync_dist_safe,
    pack_ids,
    get_sub_ses_tensors,
)
from anyBrainer.engines.factory import UnitFactory
from anyBrainer.utils.models import (
    summarize_model_params,
    init_swin_with_residual_convs,
)
from anyBrainer.schedulers.param_schedulers import ParameterScheduler
from anyBrainer.utils.data import modality_to_onehot
from anyBrainer.utils.eval import top1_accuracy


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """
    Base class based on PyTorch Lightning Module for all models.
    
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
        model_kwargs: dict[str, Any],
        loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        other_schedulers: list[dict[str, Any]] = [],
        weights_init_fn: Callable | None = None,
        ignore_hparams: list[str] = [],
    ):
        super().__init__()
        self.model = cast(nn.Module, UnitFactory.get_model_instance_from_kwargs(model_kwargs))
        self.loss_fn = cast(list[nn.Module], UnitFactory.get_loss_fn_instances_from_kwargs(loss_fn_kwargs))
        self.optimizer_kwargs = optimizer_kwargs # instantiated in configure_optimizers
        self.lr_scheduler_kwargs = lr_scheduler_kwargs # instantiated in configure_optimizers
        self.other_schedulers_step, self.other_schedulers_epoch = (
            cast(tuple[list[ParameterScheduler], list[ParameterScheduler]],
                 UnitFactory.get_param_scheduler_instances_from_kwargs(other_schedulers))
        )
        
        if weights_init_fn is not None:
            self.model.apply(weights_init_fn)
        
        self.save_hyperparameters(
            ignore=["ignore_hparams"] + ignore_hparams, logger=True
        )
    
    def configure_optimizers(
        self,
    ) -> (optim.Optimizer | dict[str, Any] | list[optim.Optimizer] | 
          tuple[list[optim.Optimizer], list[dict[str, Any]]]):
        """
        Configure optimizers and LR schedulers.
        
        Aligning with Lighnting's documentation, it can return the following:
        (depending on the optimizer_kwargs and scheduler_kwargs inputs)
        - A single optimizer
        - A dict with a single optimizer and a single lr_scheduler_config
        - A list of optimizers
        - A list of optimizers with a list of lr_scheduler_configs
        """
        optimizer = cast(optim.Optimizer | list[optim.Optimizer], 
                         UnitFactory.get_optimizer_instances_from_kwargs(self.optimizer_kwargs, self.model))

        if self.lr_scheduler_kwargs is not None:
            lr_scheduler = cast(dict[str, Any] | list[dict[str, Any]],
                                UnitFactory.get_lr_scheduler_instances_from_kwargs(self.lr_scheduler_kwargs, optimizer))
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

    def summarize(self) -> None:
        """Show model parameters."""
        logger.info(summarize_model_params(self))
    
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


class CLwAuxModel(BaseModel):
    """Contrastive learning with auxiliary loss model."""
    queue: torch.Tensor

    def __init__(
        self,
        *,
        model_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        loss_kwargs: dict[str, Any] = {},
        loss_scheduler_kwargs: dict[str, Any] = {},
        momentum_scheduler_kwargs: dict[str, Any] = {},
        weights_init_fn: Callable | str | None = init_swin_with_residual_convs,
        logits_postprocess_fn: Callable | str | None = None,
        ignore_hparams: list[str] = [],
        **kwargs,
    ):  
        loss_fn_kwargs = [
            {
                "name": "InfoNCELoss",
                "temperature": loss_kwargs.get("temperature", 0.1),
                "top_k_negatives": loss_kwargs.get("top_k_negatives", None),
            },
            {
                "name": "CrossEntropyLoss",
                "weight": loss_kwargs.get("cross_entropy_weights", None),
            },
        ]
        
        other_schedulers = [
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "loss_weight",
                "start_step": loss_scheduler_kwargs.get("loss_weight_step_start", 0),
                "end_step": loss_scheduler_kwargs.get("loss_weight_step_end", 0),
                "start_value": loss_scheduler_kwargs.get("loss_weight_start_value", 0.5),
                "end_value": loss_scheduler_kwargs.get("loss_weight_end_value", 0.5),
                "mode": "linear",
            },
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "momentum",
                "start_step": momentum_scheduler_kwargs.get("momentum_step_start", 0),
                "end_step": momentum_scheduler_kwargs.get("momentum_step_end", 0),
                "start_value": momentum_scheduler_kwargs.get("momentum_start_value", 0.999),
                "end_value": momentum_scheduler_kwargs.get("momentum_end_value", 0.999),
                "mode": "linear",
            },
        ]

        # TODO: get functions from registry

        super().__init__(
            model_kwargs=model_kwargs,
            loss_fn_kwargs=loss_fn_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            other_schedulers=other_schedulers,
            weights_init_fn=weights_init_fn, # type: ignore
            ignore_hparams=ignore_hparams,
        )
        
        # Initialize key encoder
        self.key_encoder = deepcopy(self.model)
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        
        # Initialize queue
        self.queue_size = loss_kwargs.get("queue_size", 16384)
        self.proj_dim = model_kwargs.get("proj_dim", 128)
        self.register_buffer("queue", torch.empty(0, self.proj_dim))
        self.register_buffer("queue_ids", torch.empty(0, dtype=torch.long))

        logger.info(f"\nLightning module initialized with following "
                    f"hyperparameters:\n{self.hparams}")

    @torch.no_grad()
    def _get_negatives(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Return negative features that do not share (subj, sess) IDs with
        the current batch. No mutation of the queue.
        """
        if self.queue.numel() == 0:
            return torch.empty(0, self.proj_dim, device=batch_ids.device)

        # mask out rows where batch_ids are in queue_ids
        non_dup_mask = ~torch.isin(self.queue_ids, batch_ids) # type: ignore
        return self.queue[non_dup_mask]

    @torch.no_grad()
    def _update_queue(self, q: torch.Tensor, q_ids: torch.Tensor) -> None:
        """Update queue with new embeddings and IDs."""
        q = q.detach()
        q_ids = q_ids.detach()

        if self.queue.numel() == 0:
            self.queue.resize_(q.shape).copy_(q)
            self.queue_ids.resize_(q_ids.shape).copy_(q_ids)
        else:
            self.queue = torch.cat([self.queue, q], dim=0)
            self.queue_ids = torch.cat([self.queue_ids, q_ids], dim=0)
            
        if self.queue.shape[0] > self.queue_size:
            excess = self.queue.shape[0] - self.queue_size
            self.queue = self.queue[excess:]
            self.queue_ids = self.queue_ids[excess:]

        self.log("train/queue_size", self.queue.shape[0], on_step=True, prog_bar=False, 
                sync_dist=sync_dist_safe(self))
    
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
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)
        
        self.log("train/momentum", momentum, on_step=True, prog_bar=False, 
                sync_dist=sync_dist_safe(self))

    def _compute_loss(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        queue: torch.Tensor,
        q_aux: torch.Tensor,
        aux_spr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined InfoNCE loss with auxiliary CE loss."""
        loss_info_nce, cl_stats = self.loss_fn[0](q_proj, k_proj, queue) # type: ignore
    
        loss_aux = self.loss_fn[1](q_aux, aux_spr) # type: ignore

        loss_weight = self.get_step_scheduler_values()[0]["loss_weight"]
        
        combined_loss = (loss_info_nce * loss_weight + 
                         loss_aux * (1 - loss_weight))
        
        return combined_loss, {"loss_info_nce": loss_info_nce, 
                               "loss_aux": loss_aux,
                               "loss_weight": loss_weight, 
                               **cl_stats}

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        """Get modality one-hot labels to device."""
        if dataloader_idx != 3: # not for prediction
            batch["aux_labels"] = modality_to_onehot(batch, "mod", batch["query"].device)
            batch["sub_id"], batch["ses_id"] = get_sub_ses_tensors(batch, batch["query"].device)
        return batch
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._update_key_encoder()
    
    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        # Filter positive pairs from queue
        batch_ids = pack_ids(batch["sub_id"], batch["ses_id"])
        negatives = self._get_negatives(batch_ids)

        loss, loss_dict = self._compute_loss(
            q_proj=q_proj,
            k_proj=k_proj,
            queue=negatives,
            q_aux=q_aux,
            aux_spr=batch["aux_labels"],
        )
        
        self.log_dict({
            "train/loss": loss,
            "train/loss_info_nce": loss_dict["loss_info_nce"],
            "train/loss_aux": loss_dict["loss_aux"],
            "train/loss_weight": loss_dict["loss_weight"],
            "train/pos_mean": loss_dict.get("pos_mean", torch.tensor(0.0)),
            "train/neg_mean": loss_dict.get("neg_mean", torch.tensor(0.0)),
            "train/neg_entropy": loss_dict.get("neg_entropy", torch.tensor(0.0)),
            "train/contrastive_acc": loss_dict.get("contrastive_acc", torch.tensor(0.0)),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_dist_safe(self))

        self._update_queue(q_proj, batch_ids)
            
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        loss, loss_dict = self._compute_loss(
            q_proj=q_proj,
            k_proj=k_proj,
            queue=self.queue,
            q_aux=q_aux,
            aux_spr=batch["aux_labels"],
        )
        acc = top1_accuracy(q_proj, batch["aux_labels"])
        
        self.log_dict({
            "val/loss": loss,
            "val/loss_info_nce": loss_dict["loss_info_nce"],
            "val/loss_aux": loss_dict["loss_aux"],
            "val/loss_weight": loss_dict["loss_weight"],
            "val/aux_acc": acc,
        }, on_epoch=True, prog_bar=True, sync_dist=sync_dist_safe(self))
    
    def test_step(self, batch: dict, batch_idx: int):
        """Test step."""
        q_proj, q_aux = self.model(batch["query"])
        k_proj, _ = self.key_encoder(batch["key"])

        loss, loss_dict = self._compute_loss(
            q_proj=q_proj,
            k_proj=k_proj,
            queue=self.queue,
            q_aux=q_aux,
            aux_spr=batch["aux_labels"],
        )
        acc = top1_accuracy(q_proj, batch["aux_labels"])
        
        self.log_dict({
            "test/loss": loss,
            "test/loss_info_nce": loss_dict["loss_info_nce"],
            "test/loss_aux": loss_dict["loss_aux"],
            "test/loss_weight": loss_dict["loss_weight"],
            "test/aux_acc": acc,
        }, on_epoch=True, prog_bar=True, sync_dist=sync_dist_safe(self))