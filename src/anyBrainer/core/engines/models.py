"""Base PyTorch Lightning Module to control all model-specific operations, such
as training, validation, and testing steps.

The model is responsible for:
- Defining the model architecture
- Defining the loss function
- Defining the optimizer
- Defining the learning rate scheduler
- Defining the training, validation, and testing steps
"""

__all__ = [
    "CLwAuxModel",
    "ClassificationModel",
    "RegressionModel",
    "SegmentationModel",
    "ClassificationMidFusionModel",
]

import logging
from typing import Any, Callable, Literal, cast
from copy import deepcopy
import math

import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only
from monai.metrics.metric import CumulativeIterationMetric

from anyBrainer.registry import register
from anyBrainer.core.engines.utils import (
    sync_dist_safe,
    pack_ids,
    get_sub_ses_tensors,
    dict_get_as_tensor,
    setup_all_mixins,
    scale_labels_if_needed,
    unscale_preds_if_needed,
    mask_to_image_grid,
    get_label_mid_slice,
)
from anyBrainer.core.utils import (
    summarize_model_params,
    modality_to_idx,
    top1_accuracy,
    effective_rank,
    feature_variance,
    callable_name,
    pad_to_size,
    ensure_tuple_dim,
)
from anyBrainer.config import resolve_fn, resolve_metric
from anyBrainer.registry import RegistryKind as RK
from anyBrainer.core.engines.mixins import (
    ModelInitMixin,
    LossMixin,
    ParamSchedulerMixin,
    OptimConfigMixin,
    WeightInitMixin,
    HParamsMixin,
    InfererMixin,
    ArtifactsMixin,
)

logger = logging.getLogger(__name__)


class CoreLM(pl.LightningModule):
    """Core Lightning Module that enables the use of mixins.

    Expects all configuration to be passed via kwargs and delegated to
    setup_mixin() methods defined in PLModuleMixin subclasses.
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
    ArtifactsMixin,
    CoreLM,
):
    """High-level LightningModule that wires together all standard mixins.

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
        weights_init_settings: dict[str, Any] | None = None,
        inference_settings: dict[str, Any] | None = None,
        ignore_hparams: list[str] | None = None,
        artifacts_settings: dict[str, Any] | None = None,
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
            weights_init_settings=weights_init_settings,
            inference_settings=inference_settings,
            ignore_hparams=ignore_hparams,
            artifacts_settings=artifacts_settings,
            **extra,
        )

    def configure_optimizers(self):
        """OptimConfigMixin defines get_optimizers_and_schedulers()"""
        return self.get_optimizers_and_schedulers()

    def summarize(self) -> str:
        """Show model parameters."""
        return summarize_model_params(self)

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

    queue: torch.Tensor
    queue_ids: torch.Tensor
    queue_ptr: torch.Tensor

    def __init__(
        self,
        *,
        model_kwargs: dict[str, Any],
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        loss_scheduler_kwargs: dict[str, Any] | None = None,
        momentum_scheduler_kwargs: dict[str, Any] | None = None,
        weights_init_settings: dict[str, Any] | None = None,
        inference_settings: dict[str, Any] | None = None,
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
                "start_value": loss_scheduler_kwargs.get(
                    "loss_weight_start_value", 0.5
                ),
                "end_value": loss_scheduler_kwargs.get("loss_weight_end_value", 0.5),
                "mode": "linear",
            },
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "late_ramp",
                "start_step": loss_scheduler_kwargs.get("late_ramp_step_start", 0),
                "end_step": loss_scheduler_kwargs.get("late_ramp_step_end", 0),
                "start_value": 0.0,
                "end_value": loss_scheduler_kwargs.get("late_ramp_end_value", 0.0),
                "mode": "linear",
            },
            {
                "name": "StepwiseParameterScheduler",
                "interval": "step",
                "param_name": "momentum",
                "start_step": momentum_scheduler_kwargs.get("momentum_step_start", 0),
                "end_step": momentum_scheduler_kwargs.get("momentum_step_end", 0),
                "start_value": momentum_scheduler_kwargs.get(
                    "momentum_start_value", 0.999
                ),
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
            weights_init_settings=weights_init_settings,
            inference_settings=inference_settings,
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
        self.register_buffer(
            "queue_ids", torch.full((self.queue_size,), -1, dtype=torch.long)
        )
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
            self.queue[: end % self.queue_size] = key[first:]
            self.queue_ids[ptr:] = key_ids[:first]
            self.queue_ids[: end % self.queue_size] = key_ids[first:]

        self.queue_ptr[0] = end % self.queue_size

        self.log(
            "train/queue_size",
            min(self.queue_size, ptr + batch_size),
            on_step=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    @torch.no_grad()
    def _get_negatives(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """Get valid negative features without mutating the queue."""
        valid_mask = self.queue_ids != -1
        non_dup_mask = valid_mask & (~torch.isin(self.queue_ids, batch_ids))
        return self.queue[non_dup_mask]

    @torch.no_grad()
    def _update_key_encoder(self):
        """Update key encoder."""
        momentum = self.get_step_scheduler_values()[2]["momentum"]

        params_q = list(self.model.parameters())
        params_k = list(self.key_encoder.parameters())

        if len(params_q) != len(params_k):
            msg = "Query and key encoder param counts differ."
            logger.error(msg)
            raise RuntimeError(msg)

        for param_q, param_k in zip(params_q, params_k):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

        self.log(
            "train/momentum",
            momentum,
            on_step=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    def _compute_loss(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        queue: torch.Tensor,
        q_aux: torch.Tensor,
        aux_spr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined InfoNCE loss with auxiliary CE loss."""
        loss_info_nce, cl_stats = self.loss_fn(q_proj, k_proj, queue)  # type: ignore

        if self.ce_weights is not None:
            self.ce_weights = self.ce_weights.to(q_aux.device)

        loss_aux = F.cross_entropy(q_aux, aux_spr, weight=self.ce_weights)  # type: ignore

        loss_weight = (
            self.get_step_scheduler_values()[0]["loss_weight"]
            + self.get_step_scheduler_values()[1]["late_ramp"]
        )
        combined_loss = loss_info_nce * loss_weight + loss_aux * (1 - loss_weight)

        return combined_loss, {
            "loss_info_nce": loss_info_nce,
            "loss_aux": loss_aux,
            "loss_weight": loss_weight,
            **cl_stats,
        }

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        """Get modality one-hot labels to device."""
        if dataloader_idx != 3:  # not for prediction
            batch["aux_labels"] = modality_to_idx(batch, "mod", batch["query"].device)
            batch["sub_id"], batch["ses_id"] = get_sub_ses_tensors(
                batch, batch["query"].device
            )
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
        self.log_dict(
            {
                "train/loss": loss.item(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist_safe(self),
        )
        self.log_dict(
            {
                "train/loss_info_nce": loss_dict["loss_info_nce"].item(),
                "train/loss_aux": loss_dict["loss_aux"].item(),
                "train/loss_weight": loss_dict["loss_weight"],
                "train/pos_mean": loss_dict.get("pos_mean", torch.tensor(0)).item(),
                "train/neg_mean": loss_dict.get("neg_mean", torch.tensor(0)).item(),
                "train/contrastive_acc": loss_dict.get(
                    "contrastive_acc", torch.tensor(0)
                ).item(),
                "train/eff_rank": effective_rank(negatives)[0].item(),
                "train/feature_variance": feature_variance(negatives).item(),
                "train/aux_acc": top1_accuracy(q_aux, batch["aux_labels"]).item(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

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
        self.log_dict(
            {
                "val/loss": loss.item(),
                "val/loss_info_nce": loss_dict["loss_info_nce"].item(),
                "val/loss_aux": loss_dict["loss_aux"].item(),
                "val/contrastive_acc": loss_dict["contrastive_acc"].item(),
                "val/aux_acc": top1_accuracy(q_aux, batch["aux_labels"]).item(),
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist_safe(self),
        )

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
        self.log_dict(
            {
                "test/loss": loss.item(),
                "test/loss_info_nce": loss_dict["loss_info_nce"].item(),
                "test/loss_aux": loss_dict["loss_aux"].item(),
                "test/loss_weight": loss_dict["loss_weight"],
                "test/aux_acc": top1_accuracy(q_proj, batch["aux_labels"]).item(),
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist_safe(self),
        )


@register(RK.PL_MODULE)
class MultimodalMIMModel(BaseModel):
    """Multimodal MIM-style pre-training model with a hybrid conv-transformer
    architecture."""

    def __init__(self, **base_model_kwargs) -> None:
        super().__init__(**base_model_kwargs)

        if isinstance(self.loss_fn, list):
            msg = (
                f"[{self.__class__.__name__}] expected a single loss function for MIM-style"
                f"reconstruction, but got n={len(self.loss_fn)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        def _infer_attr(name: str) -> Any:
            enc = getattr(self.model, "encoder", None)
            if enc is not None and hasattr(enc, name):
                return getattr(enc, name)
            if hasattr(self.model, name):
                return getattr(self.model, name)
            return None

        self.spatial_dims = _infer_attr("spatial_dims")
        if self.spatial_dims not in (2, 3):
            msg = f"[{self.__class__.__name__}] Cannot infer valid `spatial_dims` (2/3) from model."
            logger.error(msg)
            raise ValueError(msg)

        self.embed_dim = _infer_attr("embed_dim")
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            msg = f"[{self.__class__.__name__}] Cannot infer valid `embed_dim` from model."
            logger.error(msg)
            raise ValueError(msg)

        patch_size = _infer_attr("patch_size")
        if patch_size is None:
            msg = f"[{self.__class__.__name__}] Cannot infer `patch_size` from model."
            logger.error(msg)
            raise ValueError(msg)
        self.patch_size = ensure_tuple_dim(patch_size, self.spatial_dims)

        self.in_channels = _infer_attr("in_channels")
        if not isinstance(self.in_channels, int) or self.in_channels <= 0:
            msg = f"[{self.__class__.__name__}] Cannot infer valid `in_channels` from model."
            logger.error(msg)
            raise ValueError(msg)

        if self.embed_dim % self.in_channels != 0:
            msg = (
                f"[{self.__class__.__name__}] embed_dim ({self.embed_dim}) must be divisible by "
                f"in_channels ({self.in_channels})."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.channels_per_mod = self.embed_dim // self.in_channels

    def _expand_mask_to_embed_dim(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, *patch_grid) -> (B, embed_dim, *patch_grid),
        bool."""
        if mask.ndim != 2 + self.spatial_dims:
            msg = (
                f"[{self.__class__.__name__}] Expected `mask` to have {2 + self.spatial_dims} dims "
                f"(B, C, *patch_grid), got shape {tuple(mask.shape)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        B, C, *patch_grid = mask.shape
        if C != self.in_channels:
            msg = f"[{self.__class__.__name__}] Expected mask channels={self.in_channels}, got {C}."
            logger.error(msg)
            raise ValueError(msg)

        m = mask != 0
        m = m.unsqueeze(2)  # (B, C, 1, *pg)

        # (B, C, channels_per_mod, *pg)
        rep = (1, 1, self.channels_per_mod) + (1,) * self.spatial_dims
        m = m.repeat(*rep)

        # (B, embed_dim, *pg)
        m = m.reshape(B, self.embed_dim, *patch_grid)
        return m

    def compute_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        image_grid: tuple[int, ...],
        modalities: dict[str, list[str]],
        mode: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Compute MIM-style reconstruction loss."""
        # Patch-grid mask for the model (B, embed_dim, *patch_grid)
        m_patch = self._expand_mask_to_embed_dim(mask)

        # Image-grid mask for intensity-level loss (B, in_channels, *image_grid)
        m_img = mask_to_image_grid(
            mask,
            patch_size=cast(tuple[int, ...], self.patch_size),
            image_grid=image_grid,
        )

        # Sort modalities by channel index
        modalities_sorted = tuple(
            v for _, v in sorted(modalities.items(), key=lambda kv: int(kv[0][3:]))
        )

        out = self.model(x, mask=m_patch, modality=modalities_sorted)

        # Expect out and target to be same shape and align with `m_img`.
        if out.shape != target.shape:
            msg = (
                f"[{self.__class__.__name__}] Output/target shape mismatch: out={tuple(out.shape)} "
                f"target={tuple(target.shape)}."
            )
            logger.error(msg)
            raise ValueError(msg)
        if out.shape != m_img.shape:
            msg = (
                f"[{self.__class__.__name__}] Mask/output shape mismatch: mask={tuple(m_img.shape)} "
                f"out={tuple(out.shape)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        loss_fn = cast(Callable[..., torch.Tensor], self.loss_fn)
        loss = loss_fn(out[m_img], target[m_img])

        self.log(
            f"{mode}/loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

        if mode in ["val", "test"] or (
            mode == "train"
            and self.log_every_n_steps
            and self.global_step % self.log_every_n_steps == 0
        ):
            self.log_tensors_dict(
                {
                    f"{mode}/input": x.detach().masked_fill(m_img, 0.0),
                    f"{mode}/pred": out,
                    f"{mode}/target": target,
                }
            )

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self.compute_loss(
            x=batch["img"],
            mask=batch["mask"],
            target=batch["target"],
            image_grid=batch["img"].shape[-self.spatial_dims :],
            modalities={k: v for k, v in batch.items() if k.startswith("mod")},
            mode="train",
        )

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self.compute_loss(
            x=batch["img"],
            mask=batch["mask"],
            target=batch["target"],
            image_grid=batch["img"].shape[-self.spatial_dims :],
            modalities={k: v for k, v in batch.items() if k.startswith("mod")},
            mode="val",
        )

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self.compute_loss(
            x=batch["img"],
            mask=batch["mask"],
            target=batch["target"],
            image_grid=batch["img"].shape[-self.spatial_dims :],
            modalities={k: v for k, v in batch.items() if k.startswith("mod")},
            mode="test",
        )


@register(RK.PL_MODULE)
class ClassificationModel(BaseModel):
    """Generic model that can be used for any classification task.

    Automatically computes loss and logs specified metrics,
    as long as loss_fn/metric accept (B, 1) torch.float32 labels.

    Set `flat_labels=True` to use (B,) torch.long labels; required for
    some losses, including nn.CrossEntropyLoss.

    Supports late multimodal fusion for session-level classification, given
    that the following kwargs are included in `model_kwargs`: `late_fusion`,
    `in_channels`, `spatial_dims`. If not present, late fusion will be disabled.
    This information is crucial for reshaping the input tensor (B, P, N, *spatial_dims)
    into (B, N, P, in_channels, *spatial_dims) that is required for late fusion.
    """

    def __init__(
        self,
        *,
        metrics: (
            list[Callable]
            | list[str]
            | list[dict[str, Any]]
            | Callable
            | str
            | dict[str, Any]
            | None
        ) = ["top1_accuracy"],
        flat_labels: bool = False,
        get_label_from_seg: bool = False,  # if True, get label from segmentation mask
        seg_key: str = "seg",  # key to get segmentation mask from batch
        seg_label_thres: float = 0,  # threshold to binarize segmentation mask
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        self.late_fusion = base_model_kwargs.get("model_kwargs", {}).get(
            "late_fusion", False
        )
        self.in_channels = base_model_kwargs.get("model_kwargs", {}).get(
            "in_channels", 1
        )
        self.spatial_dims = base_model_kwargs.get("model_kwargs", {}).get(
            "spatial_dims", 3
        )
        self._fusion_w: torch.Tensor | None = None
        if self.late_fusion:
            if hasattr(self.model, "fusion_head") and hasattr(
                self.model.fusion_head, "fusion_weights"
            ):
                self._fusion_w = self.model.fusion_head.fusion_weights  # type: ignore[attr-defined]
            else:
                logger.warning(
                    f"[{self.__class__.__name__}] Late fusion is enabled but "
                    f"fusion_head.fusion_weights is not found; skipping "
                    f"logging of fusion weights."
                )

        self.metrics: list[Callable] = []
        if metrics is not None:
            if isinstance(metrics, str):
                metrics = [metrics]
            elif isinstance(metrics, dict):
                metrics = [metrics]
            elif callable(metrics):
                metrics = [metrics]
            self.metrics = [cast(Callable, resolve_metric(m)) for m in metrics]

        self.flat_labels = flat_labels

        self.get_label_from_seg = get_label_from_seg
        self.seg_key = seg_key
        self.seg_label_thres = seg_label_thres

        msg = (
            f"[{self.__class__.__name__}] Initialized with "
            f"metrics={[callable_name(m) for m in self.metrics]}, "
            f"flat_labels={self.flat_labels}"
        )
        if self.get_label_from_seg:
            msg += f", get_label_from_seg={self.get_label_from_seg}, "
            msg += f"seg_key={self.seg_key}, "
            msg += f"seg_label_thres={self.seg_label_thres}."

        logger.info(msg)

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int):
        """Get labels to device."""
        # Get labels tensor to appropriate format
        if dataloader_idx != 3:  # not for prediction
            if self.get_label_from_seg:
                sums = batch[self.seg_key].sum(dim=tuple(range(-self.spatial_dims, 0)))
                lbl = (sums > self.seg_label_thres).to(dtype=torch.float32)
                batch["label"] = (
                    lbl
                    if not self.flat_labels
                    else lbl.squeeze(-1).to(dtype=torch.long)
                )
            else:
                lbl = batch["label"]
                if torch.is_tensor(lbl):
                    lbl = lbl.to(device=batch["img"].device)
                else:
                    lbl = torch.as_tensor(lbl, device=batch["img"].device)

                if self.flat_labels:  # For nn.CrossEntropyLoss, etc.: (B,) long
                    if lbl.ndim > 1:
                        lbl = lbl.squeeze(-1)
                    lbl = lbl.to(dtype=torch.long)
                else:  # Default/regression/BCE style: (B,1) float
                    lbl = lbl.to(dtype=torch.float32)
                    if lbl.ndim == 1:
                        lbl = lbl.unsqueeze(1)
                batch["label"] = lbl

        return batch

    def compute_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss; override for more complex behavior."""
        return self.loss_fn(out, target)  # type: ignore

    @torch.no_grad()
    def compute_metrics(
        self, out: torch.Tensor, target: torch.Tensor
    ) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        stats = {}
        for m in self.metrics:
            name = getattr(m, "__name__", m.__class__.__name__)
            try:
                val = m(out, target)
            except Exception:
                logger.exception(
                    f"[{self.__class__.__name__}] Failed to compute "
                    f"metric {name}; skipping."
                )
                continue

            if isinstance(m, CumulativeIterationMetric):
                agg = m.aggregate(reduction="mean")
                if isinstance(agg, torch.Tensor):
                    stats[name] = agg.item()
                elif isinstance(agg, list):
                    for metric_name, val in zip(m.metric_name, agg):  # type: ignore[attr-defined]
                        stats[f"{name}_{metric_name}"] = val.item()
                else:
                    msg = (
                        f"[{self.__class__.__name__}.compute_metrics] Unsuppoeted metric "
                        f"aggregation type: {type(agg)}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                m.reset()
            else:
                stats[name] = val.mean().item()
        return stats

    def log_step(self, step_name: str, log_dict: dict[str, Any]) -> None:
        """Logs step statistics."""
        on_step = step_name == "train"
        self.log_dict(
            {f"{step_name}/{k}": v for k, v in log_dict.items()},
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    def training_step(self, batch: dict, batch_idx: int):
        """Training step; computes loss and metrics."""
        out = self.model(batch["img"])
        loss = self.compute_loss(out, batch["label"])
        stats = self.compute_metrics(
            cast(torch.Tensor, self.postprocess(out)), batch["label"]
        )
        stats["loss"] = loss.item()
        self.log_step("train", stats)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step; performs inference and computes metrics."""
        # patch-level
        out = self.inferer(batch["img"], self.model)
        loss = self.compute_loss(out, batch["label"])
        stats = self.compute_metrics(
            cast(torch.Tensor, self.postprocess(out)), batch["label"]
        )
        stats["loss"] = loss.item()
        self.log_step("val", stats)

        # session-level
        out = cast(torch.Tensor, self.predict(batch, invert=False))
        stats = self.compute_metrics(out, batch["label"])
        self.log_step("val_infer", stats)

        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step; performs inference and computes metrics."""
        out = cast(torch.Tensor, self.predict(batch, invert=False))
        stats = self.compute_metrics(out, batch["label"])
        self.log_step("test", stats)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs sliding window inference."""
        out = cast(torch.Tensor, self.predict(batch, invert=False))
        return out

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Logs modality weights."""
        if (
            self.global_step % 20 == 0
            and self.late_fusion
            and self._fusion_w is not None
        ):
            w = torch.softmax(self._fusion_w.detach(), dim=0).to(
                "cpu", non_blocking=True
            )
            vals = {f"train/modality_{i}": v for i, v in enumerate(w.tolist())}
            self.log_dict(
                vals,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist_safe(self),
            )

    @rank_zero_only
    def on_train_end(self) -> None:
        """Logs final modality weights."""
        if self.late_fusion and self._fusion_w is not None:
            w = torch.softmax(self._fusion_w.detach(), dim=0).to(
                "cpu", non_blocking=True
            )
            vals = {f"modality_{i}": float(v) for i, v in enumerate(w)}
            logger.info(f"Final learnable modality weights: {vals}")


@register(RK.PL_MODULE)
class RegressionModel(ClassificationModel):
    """Generic model for regression tasks.

    Extends the ClassificationModel, but with the following features:
    - `flat_labels` is set to False by default.
    - `center_labels` and `scale_labels` are supported; used for centering and
       setting the bounds for scaling the labels to [-1, 1] range.
    - `metrics` is set to [mse_score, rmse_score, mae_score, r2_score, pearsonr] by default.
    - `bias_init` is used to initialize the bias of the last layer.
       Can be set to match the mean of the labels distribution to avoid early-stage loss spikes.
    """

    def __init__(
        self,
        center_labels: str | None = None,
        scale_labels: list[float] | None = None,
        bias_init: float | None = None,
        metrics: (
            list[Callable]
            | list[str]
            | list[dict[str, Any]]
            | Callable
            | str
            | dict[str, Any]
            | None
        ) = ["rmse_score", "mae_score", "r2_score", "pearsonr"],
        **base_model_kwargs,
    ):
        super().__init__(
            flat_labels=False,
            metrics=metrics,
            **base_model_kwargs,
        )

        self.center_labels = center_labels
        self.scale_labels = scale_labels
        self.labels_meta: dict[str, Any] = {}  # For storing label statistics

        msg = (
            f"[{self.__class__.__name__}] Initialized RegressionModel with "
            f"center_labels={self.center_labels}, "
            f"scale_labels={self.scale_labels}. "
        )

        if bias_init is not None:
            is_found = False
            for name, m in reversed(list(self.model.named_modules())):
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    m.bias.data.fill_(float(bias_init))
                    msg += f"Bias of layer `{name}.bias` initialized to {bias_init}."
                    is_found = True
                    break
            if not is_found:
                msg += "Could not find a linear layer to initialize bias."
        else:
            msg += "Bias initialization disabled."
        logger.info(msg)

    def compute_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss; override for more complex behavior."""
        target, self.labels_meta = scale_labels_if_needed(
            target, self.center_labels, self.scale_labels
        )
        return self.loss_fn(out, target)  # type: ignore

    @torch.no_grad()
    def compute_metrics(
        self, out: torch.Tensor, target: torch.Tensor
    ) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        out = unscale_preds_if_needed(out, self.labels_meta)
        return super().compute_metrics(out, target)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs sliding window inference."""
        out = self.inferer(batch["img"], self.model)
        return unscale_preds_if_needed(out, self.labels_meta)


@register(RK.PL_MODULE)
class SegmentationModel(BaseModel):
    """Generic model that can be used for any segmentation task.

    If `get_uncertainty` is True, returns (mean, std) instead of mean;
    effective only when TTA is provided and enabled.
    """

    def __init__(
        self,
        *,
        metrics: (
            list[Callable]
            | list[str]
            | list[dict[str, Any]]
            | Callable
            | str
            | dict[str, Any]
            | None
        ) = [
            {"name": "DiceMetric"},
            {"name": "MeanIoU"},
            {"name": "HausdorffDistanceMetric"},
            {"name": "SurfaceDistanceMetric"},
        ],
        get_uncertainty: bool = False,  # if True, returns (mean, std)
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        self.get_uncertainty = self.tta is not None and get_uncertainty

        self.metrics: list[Callable] = []
        if metrics is not None:
            if isinstance(metrics, str):
                metrics = [metrics]
            elif isinstance(metrics, dict):
                metrics = [metrics]
            elif callable(metrics):
                metrics = [metrics]
            self.metrics = [cast(Callable, resolve_metric(m)) for m in metrics]

        logger.info(
            f"[{self.__class__.__name__}] Initialized with "
            f"metrics={[callable_name(m) for m in self.metrics]}."
        )

    def compute_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss; override for more complex behavior."""
        return self.loss_fn(out, target)  # type: ignore

    @torch.no_grad()
    def compute_metrics(
        self, out: torch.Tensor, target: torch.Tensor
    ) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        # assumes (B, C, ...)
        if out.size(1) != target.size(1):
            msg = (
                f"[{self.__class__.__name__}.compute_metrics] Unexpected output "
                f"and target shapes: {out.size()} and {target.size()}."
            )
            logger.error(msg)
            raise ValueError(msg)

        stats = {}
        for m in self.metrics:
            name = getattr(m, "__name__", m.__class__.__name__)
            try:
                val = m(out, target)
            except Exception:
                logger.exception(
                    f"[{self.__class__.__name__}] Failed to compute "
                    f"metric {name}; skipping."
                )
                continue

            if isinstance(m, CumulativeIterationMetric):
                agg = m.aggregate(reduction="mean")
                if isinstance(agg, torch.Tensor):
                    stats[name] = agg.item()
                elif isinstance(agg, list):
                    for metric_name, val in zip(m.metric_name, agg):  # type: ignore[attr-defined]
                        stats[f"{name}_{metric_name}"] = val.item()
                else:
                    msg = (
                        f"[{self.__class__.__name__}.compute_metrics] Unsuppoeted metric "
                        f"aggregation type: {type(agg)}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                m.reset()
            else:
                stats[name] = val.mean().item()
        return stats

    def log_step(self, step_name: str, log_dict: dict[str, Any]) -> None:
        """Logs step statistics."""
        on_step = step_name == "train"
        self.log_dict(
            {f"{step_name}/{k}": v for k, v in log_dict.items()},
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    def training_step(self, batch: dict, batch_idx: int):
        """Training step; computes loss and metrics."""
        out = self.model(batch["img"])
        loss = self.compute_loss(out, batch["seg"])
        stats = self.compute_metrics(
            cast(torch.Tensor, self.postprocess(out)), batch["seg"]
        )
        stats["loss"] = loss.item()
        self.log_step("train", stats)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step; performs inference and computes metrics."""
        out_no_tta = cast(torch.Tensor, self.predict(batch, do_tta=False))

        # Non-TTA stats; regular val/
        loss = self.compute_loss(out_no_tta, batch["seg"])
        stats = self.compute_metrics(out_no_tta, batch["seg"])
        stats["loss"] = loss.item()
        self.log_step("val", stats)

        # TTA stats; log mean and std
        if self.tta:
            out_tta = cast(torch.Tensor, self.predict(batch, do_tta=True))
            stats = self.compute_metrics(out_tta, batch["seg"])
            stats["loss"] = loss.item()
            self.log_step("val_tta", stats)

        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step; performs inference and computes metrics."""
        mean = cast(torch.Tensor, self.predict(batch))
        stats = self.compute_metrics(mean, batch["seg"])
        self.log_step("test", stats)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs sliding window inference."""
        return cast(torch.Tensor, self.predict(batch, return_std=self.get_uncertainty))

    def _shared_inference(
        self,
        batch: dict,
        *,
        eps: float = 1e-6,
        min_std: float = 1e-3,
        n_spatial_dims: int = 3,
        weighting: Literal["invvar", "softmin", "uniform"] = "invvar",
        softmin_tau: float = 0.1,
        accumulate_on_cpu: bool = False,
        do_tta: bool = True,
        do_postprocess: bool = True,
        return_std: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse segmentation logits across multiple images (img_*) using
        uncertainty weights. Works for shapes (B, C, H, W, D) or (B, 1, ...).
        Averages in logit space based on the standard deviation of the TTA
        outputs.

        Uses same API with `predict` method, but with additional arguments.
        """
        infer_keys = [k for k in batch if k.startswith("img_")]
        if not infer_keys:
            msg = "No 'img_*' keys found in batch."
            logger.error(msg)
            raise ValueError(msg)

        if weighting == "softmin" and softmin_tau <= 0:
            msg = "`softmin_tau` must be > 0"
            logger.error(msg)
            raise ValueError(msg)

        # Decide accumulation device
        acc_device = (
            torch.device("cpu") if accumulate_on_cpu else batch[infer_keys[0]].device
        )

        # Determine spatial_dims and target (max) spatial size from inputs
        target_spatial = list(batch[infer_keys[0]].shape[-n_spatial_dims:])
        for k in infer_keys[1:]:
            shp = batch[k].shape[-n_spatial_dims:]
            target_spatial = [max(a, b) for a, b in zip(target_spatial, shp)]

        def _pad(x: torch.Tensor) -> torch.Tensor:
            return pad_to_size(
                x, target_spatial, n_spatial_dims, mode="constant", side="both"
            )

        # Accumulators
        w_sum: torch.Tensor | None = None  # Σ w_i
        w_mu_sum: torch.Tensor | None = None  # Σ w_i * μ_i
        w_mu2_sum: torch.Tensor | None = None  # Σ w_i * μ_i^2 (if return_std)
        w_var_sum: torch.Tensor | None = None  # Σ w_i * σ_i^2 (if return_std)

        def _alloc(full_like: torch.Tensor, need_var: bool) -> None:
            nonlocal w_sum, w_mu_sum, w_mu2_sum, w_var_sum
            full_like = full_like.to(acc_device)
            w_sum = torch.zeros_like(full_like, device=acc_device)
            w_mu_sum = torch.zeros_like(full_like, device=acc_device)
            if need_var:
                w_mu2_sum = torch.zeros_like(full_like, device=acc_device)
                w_var_sum = torch.zeros_like(full_like, device=acc_device)

        # Aggregate logits across multiple images (img_*)
        if not self.tta or not do_tta:
            # Uniform weighting; std not available
            if return_std:
                return_std = False  # std can't be computed without TTA

            for idx, k in enumerate(infer_keys):
                out = cast(
                    torch.Tensor,
                    self.predict(
                        batch, img_key=k, return_std=False, do_postprocess=False
                    ),
                )
                if accumulate_on_cpu:
                    out = out.cpu()
                out = _pad(out)

                if idx == 0:
                    _alloc(out, need_var=False)

                w_sum = cast(torch.Tensor, w_sum)
                w_mu_sum = cast(torch.Tensor, w_mu_sum)

                w_sum += 1.0
                w_mu_sum += out

        else:
            for idx, k in enumerate(infer_keys):
                mu_i, std_i = self.predict(
                    batch, img_key=k, return_std=True, do_postprocess=False, do_tta=True
                )

                std_i = torch.nan_to_num(
                    std_i, nan=min_std, posinf=min_std, neginf=min_std
                ).clamp_min(min_std)
                var_i = std_i * std_i + eps

                if weighting == "invvar":
                    w_i = 1.0 / var_i
                elif weighting == "softmin":
                    w_i = torch.exp(-std_i / softmin_tau)
                elif weighting == "uniform":
                    w_i = torch.ones_like(var_i)
                else:
                    raise ValueError(f"Unknown weighting '{weighting}'")

                if accumulate_on_cpu:
                    mu_i, var_i, w_i = mu_i.cpu(), var_i.cpu(), w_i.cpu()

                mu_i = _pad(mu_i)
                var_i = _pad(var_i)
                w_i = _pad(w_i)

                if idx == 0:
                    _alloc(mu_i, need_var=return_std)

                # Accumulate
                w_sum = cast(torch.Tensor, w_sum)
                w_mu_sum = cast(torch.Tensor, w_mu_sum)
                w_sum += w_i
                w_mu_sum += w_i * mu_i

                if return_std:
                    w_mu2_sum = cast(torch.Tensor, w_mu2_sum)
                    w_var_sum = cast(torch.Tensor, w_var_sum)
                    w_mu2_sum += w_i * (mu_i * mu_i)
                    w_var_sum += w_i * var_i

                del mu_i, std_i, var_i, w_i

        # Compute stats and postprocess
        w_sum = cast(torch.Tensor, w_sum)
        w_mu_sum = cast(torch.Tensor, w_mu_sum)
        mu = w_mu_sum / (w_sum + 1e-12)

        if return_std:
            w_mu2_sum = cast(torch.Tensor, w_mu2_sum)
            w_var_sum = cast(torch.Tensor, w_var_sum)
            second_moment = (w_var_sum + w_mu2_sum) / (w_sum + 1e-12)
            var_pred = (second_moment - mu * mu).clamp_min(0)
            std_pred = torch.sqrt(var_pred)

        out = cast(torch.Tensor, self.postprocess(mu)) if do_postprocess else mu
        return (out, std_pred) if return_std else out


@register(RK.PL_MODULE)
class ClassificationMidFusionModel(BaseModel):
    """Classification model that fuses multiple modalities using mid-fusion.
    Assumes that a mid-fusion layer is present in the model.

    Assumes that the model is a `Swinv2ClassifierMidFusion` class, which
    performs mid-fusion of multiple modalities, before aggregating the spatial
    dimensions of the feature maps.

    Proceeds to perform patch-level classification.
    """

    def __init__(
        self,
        *,
        metrics: (
            list[Callable]
            | list[str]
            | list[dict[str, Any]]
            | Callable
            | str
            | dict[str, Any]
            | None
        ) = ["top1_accuracy"],
        flat_labels: bool = False,
        label_thres: int = 0,
        **base_model_kwargs,
    ):
        """
        Args:
            metrics: Metrics to compute.
            flat_labels: If True, use (B,) torch.long labels; required for
                some losses, including nn.CrossEntropyLoss.
            label_thres: Threshold for binarizing labels.

        See `BaseModel` for expected `base_model_kwargs`.
        """
        super().__init__(**base_model_kwargs)

        # Init score head
        if not hasattr(self.model, "score_head"):
            msg = (
                f"[{self.__class__.__name__}] Model does not have a "
                f"`score_head`; cannot use mid-fusion."
            )
            logger.error(msg)
            raise ValueError(msg)
        torch.nn.init.zeros_(self.model.score_head.weight)  # type: ignore[attr-defined]
        with torch.no_grad():
            self.model.score_head.bias.fill_(math.log(1e-3) - math.log1p(-1e-3))  # type: ignore[attr-defined]

        # Get fusion weights
        if not hasattr(self.model, "fusion_weights"):
            msg = (
                f"[{self.__class__.__name__}] Model does not have a "
                f"`fusion_weights`; cannot use mid-fusion."
            )
            logger.error(msg)
            raise ValueError(msg)
        self._fusion_w: torch.Tensor = cast(torch.Tensor, self.model.fusion_weights)  # type: ignore[attr-defined]

        # Get spatial dims; used for label binarization
        if hasattr(self.model, "spatial_dims"):
            self.spatial_dims = cast(int, self.model.spatial_dims)
        else:
            logger.warning(
                f"[{self.__class__.__name__}] Spatial dimensions not found in model; "
                f"assuming 3D."
            )
            self.spatial_dims = 3

        # Init metrics
        self.metrics: list[Callable] = []
        if metrics is not None:
            if isinstance(metrics, str):
                metrics = [metrics]
            elif isinstance(metrics, dict):
                metrics = [metrics]
            elif callable(metrics):
                metrics = [metrics]
            self.metrics = [cast(Callable, resolve_metric(m)) for m in metrics]

        self.label_thres = label_thres
        self.flat_labels = flat_labels

        logger.info(
            f"[{self.__class__.__name__}] Initialized with "
            f"metrics={[callable_name(m) for m in self.metrics]}, "
            f"flat_labels={self.flat_labels}."
        )

    def on_after_batch_transfer(
        self, batch: dict, dataloader_idx: int
    ) -> dict[str, Any]:
        """Get input tensor to appropriate shape and labels to device.

        Expects batch to at least have:
        - img: (B, N, *spatial_dims)
        - seg: (B, *spatial_dims)
        """
        # Get labels tensor to appropriate format
        sums = batch["seg"].sum(dim=tuple(range(-self.spatial_dims, 0)))
        lbl = (sums > self.label_thres).to(dtype=torch.float32)
        batch["label"] = (
            lbl if not self.flat_labels else lbl.squeeze(-1).to(dtype=torch.long)
        )

        return batch

    def compute_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss; override for more complex behavior."""
        return self.loss_fn(out, target)  # type: ignore

    @torch.no_grad()
    def compute_metrics(
        self, out: torch.Tensor, target: torch.Tensor
    ) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        stats = {}
        for m in self.metrics:
            name = getattr(m, "__name__", m.__class__.__name__)
            try:
                val = m(out, target)
            except Exception:
                logger.exception(
                    f"[{self.__class__.__name__}] Failed to compute "
                    f"metric {name}; skipping."
                )
                continue

            if isinstance(m, CumulativeIterationMetric):
                agg = m.aggregate(reduction="mean")
                if isinstance(agg, torch.Tensor):
                    stats[name] = agg.item()
                elif isinstance(agg, list):
                    for metric_name, val in zip(m.metric_name, agg):  # type: ignore[attr-defined]
                        stats[f"{name}_{metric_name}"] = val.item()
                else:
                    msg = (
                        f"[{self.__class__.__name__}.compute_metrics] Unsuppoeted metric "
                        f"aggregation type: {type(agg)}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                m.reset()
            else:
                stats[name] = val.mean().item()
        return stats

    def log_step(self, step_name: str, log_dict: dict[str, Any]) -> None:
        """Logs step statistics."""
        on_step = step_name == "train"
        self.log_dict(
            {f"{step_name}/{k}": v for k, v in log_dict.items()},
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    def training_step(self, batch: dict, batch_idx: int):
        """Training step; computes loss and metrics."""
        out = self.model(batch["img"])
        loss = self.compute_loss(out, batch["label"])
        stats = self.compute_metrics(
            cast(torch.Tensor, self.postprocess(out)), batch["label"]
        )
        stats["loss"] = loss.item()
        self.log_step("train", stats)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step; performs inference and computes metrics."""
        out = self.predict(batch, invert=False)
        stats = self.compute_metrics(cast(torch.Tensor, out), batch["label"])
        self.log_step("val", stats)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step; performs inference and computes metrics."""
        out = self.predict(batch, invert=False)
        stats = self.compute_metrics(cast(torch.Tensor, out), batch["label"])
        self.log_step("test", stats)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs inference."""
        return cast(torch.Tensor, self.predict(batch, invert=False))

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Logs modality weights."""
        if self.global_step % 20 == 0:
            w = torch.softmax(self._fusion_w.detach(), dim=0).to(
                "cpu", non_blocking=True
            )
            vals = {f"train/modality_{i}": v for i, v in enumerate(w.tolist())}
            self.log_dict(
                vals,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist_safe(self),
            )


@register(RK.PL_MODULE)
class MultimodalDownstreamModel(BaseModel):
    """Multimodal downstream model that can be used for any multimodal
    downstream task."""

    def __init__(
        self,
        *,
        metrics: (
            list[Callable]
            | list[str]
            | list[dict[str, Any]]
            | Callable
            | str
            | dict[str, Any]
            | None
        ),
        modality_remap: dict[str, str] | None = None,
        fallback_modality: str | None = None,
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        self.metrics: list[Callable] = []
        if metrics is not None:
            if isinstance(metrics, str):
                metrics = [metrics]
            elif isinstance(metrics, dict):
                metrics = [metrics]
            elif callable(metrics):
                metrics = [metrics]
            self.metrics = [cast(Callable, resolve_metric(m)) for m in metrics]

        self.mod_remap = modality_remap
        if self.modality_remap is not None:
            self.modality_remap = {
                str(k): str(v) for k, v in self.modality_remap.items()
            }

        self.fallback_modality = (
            str(fallback_modality) if fallback_modality is not None else None
        )

        msg = (
            f"[{self.__class__.__name__}] Initialized with "
            f"metrics={[callable_name(m) for m in self.metrics]}"
        )
        if self.modality_remap is not None:
            msg += f", modality_remap={self.modality_remap}"
            msg += f", fallback '{self.fallback_modality}'"
        msg += "."

        logger.info(msg)

    def compute_loss(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes loss; override for more complex behavior."""
        if isinstance(self.loss_fn, list):
            msg = (
                f"[{self.__class__.__name__}] Multiple loss functions not supported; "
                f"override `compute_loss` for more complex behavior."
            )
            logger.error(msg)
            raise ValueError(msg)
        return self.loss_fn(out, target)  # type: ignore[attr-defined]

    @torch.no_grad()
    def compute_metrics(
        self, out: torch.Tensor, target: torch.Tensor
    ) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        stats = {}
        for m in self.metrics:
            name = getattr(m, "__name__", m.__class__.__name__)
            try:
                val = m(out, target)
            except Exception:
                logger.exception(
                    f"[{self.__class__.__name__}] Failed to compute "
                    f"metric {name}; skipping."
                )
                continue

            if isinstance(m, CumulativeIterationMetric):
                agg = m.aggregate(reduction="mean")
                if isinstance(agg, torch.Tensor):
                    stats[name] = agg.item()
                elif isinstance(agg, list):
                    for metric_name, val in zip(m.metric_name, agg):  # type: ignore[attr-defined]
                        stats[f"{name}_{metric_name}"] = val.item()
                else:
                    msg = (
                        f"[{self.__class__.__name__}.compute_metrics] Unsuppoeted metric "
                        f"aggregation type: {type(agg)}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                m.reset()
            else:
                stats[name] = val.mean().item()
        return stats

    def log_step(self, step_name: str, log_dict: dict[str, Any]) -> None:
        """Logs step statistics."""
        on_step = step_name == "train"
        self.log_dict(
            {f"{step_name}/{k}": v for k, v in log_dict.items()},
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=sync_dist_safe(self),
        )

    def _parse_modalities(self, batch: dict) -> tuple[str, ...]:
        """Parses modalities from batch."""
        mods = {k: v for k, v in batch.items() if k.startswith("mod")}

        if self.modality_remap is not None:
            remapped = {}
            for k, v in mods.items():
                if v in self.modality_remap:
                    remapped[k] = self.modality_remap[v]
                elif self.fallback_modality is not None:
                    remapped[k] = self.fallback_modality
                else:
                    msg = (
                        f"[{self.__class__.__name__}] Modality '{v}' not found in "
                        f"`modality_remap` and no fallback provided."
                    )
                    logger.error(msg)
                    raise KeyError(msg)
            mods = remapped

        # Sort by channel index
        return tuple(v for _, v in sorted(mods.items(), key=lambda kv: int(kv[0][3:])))

    def training_step(self, batch: dict, batch_idx: int):
        """Training step; computes loss and metrics."""
        out = self.model(batch["img"], modality=self._parse_modalities(batch))
        loss = self.compute_loss(out, batch["label"])
        out = cast(torch.Tensor, self.postprocess(out))
        stats = self.compute_metrics(out, batch["label"])
        stats["loss"] = loss.item()

        self.log_step("train", stats)
        if self.log_every_n_steps and self.global_step % self.log_every_n_steps == 0:
            self.log_tensors_dict(
                {
                    "train/input": torch.where(
                        batch["label"].bool(),
                        batch["img"].max() * 1.5,
                        batch["img"],
                    ),
                    "train/pred": out,
                },
                dim=-1,
                slice_idx=get_label_mid_slice(batch["label"][0], dim=-1),
            )

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step; performs inference and computes metrics."""
        out = cast(
            torch.Tensor,
            self.predict(batch, invert=False, modality=self._parse_modalities(batch)),
        )
        stats = self.compute_metrics(out, batch["label"])
        self.log_step("val", stats)

        if batch_idx < 2:
            self.log_tensors_dict(
                {
                    "val/input": torch.where(
                        batch["label"].bool(),
                        batch["img"].max() * 1.5,
                        batch["img"],
                    ),
                    "val/pred": out,
                },
                dim=-1,
                slice_idx=get_label_mid_slice(batch["label"][0], dim=-1),
            )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Test step; performs inference and computes metrics."""
        out = cast(
            torch.Tensor,
            self.predict(batch, invert=False, modality=self._parse_modalities(batch)),
        )
        stats = self.compute_metrics(out, batch["label"])
        self.log_step("test", stats)

        if batch_idx < 2:
            self.log_tensors_dict(
                {
                    "test/input": torch.where(
                        batch["label"].bool(),
                        batch["img"].max() * 1.5,
                        batch["img"],
                    ),
                    "test/pred": out,
                },
                dim=-1,
                slice_idx=get_label_mid_slice(batch["label"][0], dim=-1),
            )

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs inference."""
        return cast(
            torch.Tensor,
            self.predict(batch, invert=False, modality=self._parse_modalities(batch)),
        )
