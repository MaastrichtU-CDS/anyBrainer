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
    "CLwAuxModel",  
]

import logging
from typing import Any, Callable
from copy import deepcopy

import torch
import lightning.pytorch as pl
import torch.nn.functional as F

from anyBrainer.registry import register
from anyBrainer.core.engines.utils import (
    sync_dist_safe,
    pack_ids,
    get_sub_ses_tensors,
    dict_get_as_tensor,
    resolve_fn,
    setup_all_mixins,
)
from anyBrainer.core.utils import (
    summarize_model_params,
    modality_to_idx,
    top1_accuracy,  
    effective_rank,
    feature_variance,
)
from anyBrainer.registry import RegistryKind as RK
from anyBrainer.core.engines.mixins import (
    ModelInitMixin,
    LossMixin,
    ParamSchedulerMixin,
    OptimConfigMixin,
    WeightInitMixin,
    HParamsMixin,
    InfererMixin,
)
from anyBrainer.core.inferers import (
    SlidingWindowClassificationInferer,
)

logger = logging.getLogger(__name__)


class CoreLM(pl.LightningModule):
    """
    Core Lightning Module that enables the use of mixins.

    Expects all configuration to be passed via kwargs and delegated
    to setup_mixin() methods defined in PLModuleMixin subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        setup_all_mixins(self, *args, **kwargs)


class BaseModel(
    ModelInitMixin,
    WeightInitMixin,
    LossMixin,
    OptimConfigMixin,
    ParamSchedulerMixin,
    InfererMixin,
    HParamsMixin,
    CoreLM,
):
    """
    High-level LightningModule that wires together all standard mixins.

    Required args:
        model_kwargs: model initialization arguments
        loss_fn_kwargs: loss function parameters; can be multiple
        optimizer_kwargs: optimizer parameters; can be multiple

    Optional kwargs (pass through to the relevant mixin, default = None):
        lr_scheduler_kwargs: learning rate scheduler parameters; can be multiple
        param_scheduler_kwargs: non-optimizer schedulers; can be multiple
        weights_init_kwargs: weight initialization parameters; includes checkpoint
        ignore_hparams: list of hyperparameters to ignore when saving ckpt
    """

    def __init__(
        self,
        *,
        model_kwargs: dict[str, Any],
        loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        param_scheduler_kwargs: list[dict[str, Any]] | None = None,
        weights_init_kwargs: dict[str, Any] | None = None,
        inferer_kwargs: dict[str, Any] | None = None,
        ignore_hparams: list[str] | None = None,
        **extra,
    ):
        if not model_kwargs:
            msg = "`model_kwargs` must be supplied and non-empty."
            logger.error(msg)
            raise ValueError(msg)
        if not loss_fn_kwargs:
            msg = "`loss_fn_kwargs` must be supplied and non-empty."
            logger.error(msg)
            raise ValueError(msg)
        if not optimizer_kwargs:
            msg = "`optimizer_kwargs` must be supplied and non-empty."
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(
            model_kwargs=model_kwargs,
            loss_fn_kwargs=loss_fn_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            param_scheduler_kwargs=param_scheduler_kwargs,
            weights_init_kwargs=weights_init_kwargs,
            inferer_kwargs=inferer_kwargs,
            ignore_hparams=ignore_hparams,
            **extra,
        )

    def configure_optimizers(self):
        """OptimConfigMixin defines get_optimizers_and_schedulers()"""
        return self.get_optimizers_and_schedulers()

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


@register(RK.PL_MODULE)
class CLwAuxModel(BaseModel):
    """Contrastive learning with auxiliary loss model."""
    queue:      torch.Tensor
    queue_ids:  torch.Tensor
    queue_ptr:  torch.Tensor

    def __init__(
        self,
        *,
        model_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        loss_scheduler_kwargs: dict[str, Any] | None = None,
        momentum_scheduler_kwargs: dict[str, Any] | None = None,
        weights_init_kwargs: dict[str, Any] | None = None,
        inferer_kwargs: dict[str, Any] | None = None,
        logits_postprocess_fn: Callable | str | None = None,
    ):  
        if loss_kwargs is None:
            loss_kwargs = {}

        if loss_scheduler_kwargs is None:
            loss_scheduler_kwargs = {}

        if momentum_scheduler_kwargs is None:
            momentum_scheduler_kwargs = {}

        loss_fn_kwargs = {
                "name": "InfoNCELoss",
                "temperature": loss_kwargs.get("temperature", 0.1),
                "top_k_negatives": loss_kwargs.get("top_k_negatives"),
                "postprocess_fn": resolve_fn(logits_postprocess_fn),
        }
        self.ce_weights = dict_get_as_tensor(loss_kwargs.get("cross_entropy_weights"))
        
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
        
        ignore_hparams = ["loss_fn_kwargs", "other_schedulers", "weights_init_kwargs"]
        if not isinstance(logits_postprocess_fn, str):
            ignore_hparams.append("logits_postprocess_fn")

        super().__init__(
            model_kwargs=model_kwargs,
            loss_fn_kwargs=loss_fn_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            param_scheduler_kwargs=other_schedulers,
            weights_init_kwargs=weights_init_kwargs,
            inferer_kwargs=inferer_kwargs,
            ignore_hparams=ignore_hparams,
        )
       
        # Initialize key encoder
        self.key_encoder = deepcopy(self.model)
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        
        # Initialize queue
        self.queue_size = loss_kwargs.get("queue_size", 16384)
        self.proj_dim = model_kwargs.get("proj_dim", 128)
        self.register_buffer("queue", torch.zeros(self.queue_size, self.proj_dim))
        self.register_buffer("queue_ids", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_queue(self, key: torch.Tensor, key_ids: torch.Tensor) -> None:
        """Update queue with new embeddings and IDs using a circular buffer."""
        key = key.detach()
        batch_size = key.size(0)
        ptr = int(self.queue_ptr)

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end] = key
            self.queue_ids[ptr:end] = key_ids
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = key[:first]
            self.queue[:end % self.queue_size] = key[first:]
            self.queue_ids[ptr:] = key_ids[:first]
            self.queue_ids[:end % self.queue_size] = key_ids[first:]

        self.queue_ptr[0] = end % self.queue_size

        self.log("train/queue_size", min(self.queue_size, ptr + batch_size), 
             on_step=True, prog_bar=False, sync_dist=sync_dist_safe(self))
    
    @torch.no_grad()
    def _get_negatives(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """Get valid negative features without mutating the queue."""
        valid_mask   = self.queue_ids != -1
        non_dup_mask = valid_mask & (~torch.isin(self.queue_ids, batch_ids))
        return self.queue[non_dup_mask]
    
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
        loss_info_nce, cl_stats = self.loss_fn(q_proj, k_proj, queue) # type: ignore

        if self.ce_weights is not None:
            self.ce_weights = self.ce_weights.to(q_aux.device)
        
        loss_aux = F.cross_entropy(q_aux, aux_spr, weight=self.ce_weights) # type: ignore

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
            batch["aux_labels"] = modality_to_idx(batch, "mod", batch["query"].device)
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
            "train/loss": loss.item(),
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=sync_dist_safe(self))
        
        self.log_dict({
            "train/loss_info_nce": loss_dict["loss_info_nce"].item(),
            "train/loss_aux": loss_dict["loss_aux"].item(),
            "train/loss_weight": loss_dict["loss_weight"],
            "train/pos_mean": loss_dict.get("pos_mean", torch.tensor(0)).item(),
            "train/neg_mean": loss_dict.get("neg_mean", torch.tensor(0)).item(),
            "train/contrastive_acc": loss_dict.get("contrastive_acc", torch.tensor(0)).item(),
            "train/eff_rank": effective_rank(negatives)[0].item(),
            "train/feature_variance": feature_variance(negatives).item(),
            "train/aux_acc": top1_accuracy(q_aux, batch["aux_labels"]).item(),
        }, on_step=True, on_epoch=True, prog_bar=False, sync_dist=sync_dist_safe(self))

        self._update_queue(k_proj, batch_ids)
            
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
        
        self.log_dict({
            "val/loss": loss.item(),
            "val/loss_info_nce": loss_dict["loss_info_nce"].item(),
            "val/loss_aux": loss_dict["loss_aux"].item(),
            "val/loss_weight": loss_dict["loss_weight"],
            "val/aux_acc": top1_accuracy(q_aux, batch["aux_labels"]).item(),
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
                
        self.log_dict({
            "test/loss": loss.item(),
            "test/loss_info_nce": loss_dict["loss_info_nce"].item(),
            "test/loss_aux": loss_dict["loss_aux"].item(),
            "test/loss_weight": loss_dict["loss_weight"],
            "test/aux_acc": top1_accuracy(q_proj, batch["aux_labels"]).item(),
        }, on_epoch=True, prog_bar=True, sync_dist=sync_dist_safe(self))


@register(RK.PL_MODULE)
class ClassificationModel(BaseModel):
    """Classification model."""
    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        """Get modality one-hot labels to device."""
        if dataloader_idx != 3: # not for prediction
            batch["label"] = torch.tensor(
                batch["label"], dtype=torch.long, device=batch["img"].device
            )
        return batch
    
    def _shared_eval_step(self, out: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict[str, Any]]:
        """Shared evaluation step."""
        loss = self.loss_fn(out, batch["label"].unsqueeze(1).float()) # type: ignore
        return loss, {"loss": loss.item(), "acc": top1_accuracy(out, batch["label"]).item()}
    
    def _log_step(self, step_name: str, log_dict: dict[str, Any]) -> None:
        """Log step statistics"""
        self.log_dict({f"{step_name}/{k}": v for k, v in log_dict.items()}, 
                      on_step=True, on_epoch=True, prog_bar=False, 
                      sync_dist=sync_dist_safe(self))

    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        out = self.model(batch["img"])
        loss, stats = self._shared_eval_step(out, batch)
        self._log_step("train", stats)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        out = self.inferer(batch["img"], self.model)
        loss, stats = self._shared_eval_step(out, batch)
        self._log_step("val", stats)
        return loss
    
    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Test step; performs sliding window inference and computes top-1 accuracy.
        """
        out = self.inferer(batch["img"], self.model)
        self._log_step("test", {"acc": top1_accuracy(out, batch["label"]).item()})
    
    def predict_step(self, batch: dict, batch_idx: int):
        """Predict step; performs sliding window inference."""
        out = self.inferer(batch["img"], self.model)
        return out