"""Mixin classes for Pytorch Lightning modules."""

from __future__ import annotations

__all__ = [
    "ModelInitMixin",
    "LossMixin",
    "ParamSchedulerMixin",
    "OptimConfigMixin",
    "WeightInitMixin",
    "HParamsMixin",
    "ArtifactsMixin",
]

import logging
from typing import Any, Callable, TYPE_CHECKING, cast
from copy import deepcopy

import torch
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch import LightningModule as plModule
from monai.inferers.inferer import Inferer

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    Identity,
    Transform,
)

from anyBrainer.core.utils import (
    load_param_group_from_ckpt,
    resolve_path,
    callable_name,
    get_parameter_groups_from_prefixes,
    split_decay_groups_from_params,
    apply_tta_monai_batch,
    to_uint8_image,
)
from anyBrainer.config import resolve_fn, resolve_transform
from anyBrainer.core.engines.utils import (
    format_optimizer_log,
)
from anyBrainer.factories import UnitFactory
from anyBrainer.interfaces import PLModuleMixin
from anyBrainer.registry import register, RegistryKind as RK

if TYPE_CHECKING:
    import torch.optim as optim
    from torch.nn import Module as nnModule
    from anyBrainer.interfaces import ParameterScheduler

logger = logging.getLogger(__name__)


@register(RK.PL_MODULE_MIXIN)
class ModelInitMixin(PLModuleMixin):
    """Adds support for instantiating the model."""

    model: nnModule

    def setup_mixin(
        self,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> None:
        """Create model instance using the dedicated factory."""
        if not isinstance(model_kwargs, dict):
            msg = (
                f"[{self.__class__.__name__}] `model_kwargs` must be a dict, "
                f"got {type(model_kwargs).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if hasattr(self, "model"):
            logger.warning(
                f"[{self.__class__.__name__}] Attribute `model' already exists "
                "and will be overwritten."
            )

        self.model = UnitFactory.get_model_instance_from_kwargs(model_kwargs)

        logger.info(
            f"[{self.__class__.__name__}] Instantiated model: {model_kwargs['name']}."
        )


@register(RK.PL_MODULE_MIXIN)
class WeightInitMixin(PLModuleMixin):
    """Adds support for initializing model weights."""

    def setup_mixin(
        self,
        weights_init_settings: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """Initialize model weights either through a pretrained checkpoint or a
        custom weight initialization function."""
        if not isinstance(weights_init_settings, (dict, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `weights_init_settings` must be a dict, "
                f"or None, got {type(weights_init_settings).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, "model"):
            msg = "Expected attribute 'model' before calling WeightInitMixin.setup_mixin()."
            logger.error(msg)
            raise ValueError(msg)

        weights_init_settings = weights_init_settings or {}

        weights_init_fn = weights_init_settings.get("weights_init_fn")
        load_pretrain_weights = weights_init_settings.get("load_pretrain_weights")
        load_param_group_prefix = weights_init_settings.get("load_param_group_prefix")
        rename_map = weights_init_settings.get("rename_map")
        strict_load = weights_init_settings.get("strict_load", False)
        extra_load_kwargs = weights_init_settings.get("extra_load_kwargs", {})

        if weights_init_fn is None and load_pretrain_weights is None:
            logger.info(
                f"[{self.__class__.__name__}] Model initialized with random weights."
            )
            return

        if weights_init_fn is not None:
            self._apply_weights_init_fn(
                weights_init_fn=weights_init_fn,
            )
        if load_pretrain_weights is not None:
            if weights_init_fn is not None:
                logger.warning(
                    f"[{self.__class__.__name__}] Provided both "
                    f"`weights_init_fn` and `load_pretrain_weights`; "
                    f"the latter can override weights initialized by "
                    f"`{weights_init_fn}`."
                )
            self._load_pretrain_weights(
                load_pretrain_weights=load_pretrain_weights,
                load_param_group_prefix=load_param_group_prefix,
                rename_map=rename_map,
                strict=strict_load,
                extra_load_kwargs=extra_load_kwargs,
            )

    def _load_pretrain_weights(
        self,
        load_pretrain_weights: str,
        load_param_group_prefix: str | list[str] | None,
        rename_map: dict[str, str] | None = None,
        strict: bool = False,
        extra_load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Load pretrained weights from checkpoint.

        Allows selective loading of specific parameter groups.
        """
        try:
            self.model, stats = load_param_group_from_ckpt(
                model_instance=self.model,
                checkpoint_path=resolve_path(load_pretrain_weights),
                select_prefixes=load_param_group_prefix,
                rename_map=rename_map,
                strict=strict,
                torch_load_kwargs=extra_load_kwargs,
            )
            logger.info(
                f"[{self.__class__.__name__}] Loaded pretrained weights from checkpoint "
                f"{load_pretrain_weights}\n#### Summary ####"
                f"\n- Loaded {len(stats['loaded_keys'])} parameters "
                f"(from {len(stats['loaded_keys']) + len(stats['ignored_keys'])} parameters)"
                f"\n- Unexpected {len(stats['unexpected_keys'])} parameters"
                f"\n- Missing {len(stats['missing_keys'])} parameters"
            )
        except Exception as e:
            logger.error(
                f"[{self.__class__.__name__}] Failed to load pretrained weights: {e}"
            )
            raise e

    def _apply_weights_init_fn(
        self,
        weights_init_fn: Callable | str,
    ) -> None:
        """Apply weight initialization function to model."""
        try:
            self.model.apply(cast(Callable, resolve_fn(weights_init_fn)))
            logger.info(
                f"[{self.__class__.__name__}] Initialized model weights with "
                f"`{weights_init_fn}`."
            )
        except Exception as e:
            logger.error(
                f"[{self.__class__.__name__}] Failed to apply weight initialization function: {e}"
            )
            raise e


@register(RK.PL_MODULE_MIXIN)
class LossMixin(PLModuleMixin):
    """Adds support for instantiating the loss function(s)."""

    loss_fn: nnModule | list[nnModule]

    def setup_mixin(
        self,
        loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
        **kwargs,
    ) -> None:
        """Create single or list of loss fn instances using the dedicated
        factory."""
        if not isinstance(loss_fn_kwargs, (list, dict)):
            msg = (
                f"[{self.__class__.__name__}] `loss_fn_kwargs` must be a dict, "
                f"or list, got {type(loss_fn_kwargs).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if hasattr(self, "loss_fn"):
            logger.warning(
                f"[{self.__class__.__name__}] Attribute `loss_fn' already exists "
                "and will be overwritten."
            )

        self.loss_fn = UnitFactory.get_loss_fn_instances_from_kwargs(loss_fn_kwargs)

        if isinstance(self.loss_fn, list):
            loss_fn_names = [loss_fn.__class__.__name__ for loss_fn in self.loss_fn]
            logger.info(
                f"[{self.__class__.__name__}] Instantiated loss function(s): {loss_fn_names}."
            )
        else:
            logger.info(
                f"[{self.__class__.__name__}] Instantiated loss function: {self.loss_fn.__class__.__name__}."
            )


@register(RK.PL_MODULE_MIXIN)
class ParamSchedulerMixin(PLModuleMixin):
    """Adds support for extra (non-optimizer) schedulers; e.g., contrastive
    learning momentum, loss function weighting, etc."""

    other_schedulers_step: list[ParameterScheduler]
    other_schedulers_epoch: list[ParameterScheduler]

    def setup_mixin(
        self,
        param_scheduler_kwargs: list[dict[str, Any]] | None,
        **kwargs,
    ) -> None:
        """Initialize custom step- and epoch-based parameter schedulers."""
        self.other_schedulers_step = []
        self.other_schedulers_epoch = []

        if not isinstance(param_scheduler_kwargs, (list, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `param_scheduler_kwargs` must be a list or None, "
                f"got {type(param_scheduler_kwargs).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if param_scheduler_kwargs is not None:
            self.other_schedulers_step, self.other_schedulers_epoch = (
                UnitFactory.get_param_scheduler_instances_from_kwargs(
                    param_scheduler_kwargs
                )
            )
            step_scheduler_names = [
                scheduler.__class__.__name__ for scheduler in self.other_schedulers_step
            ]
            epoch_scheduler_names = [
                scheduler.__class__.__name__
                for scheduler in self.other_schedulers_epoch
            ]

            logger.info(
                f"[{self.__class__.__name__}] Instantiated {len(step_scheduler_names)} "
                f"step-wise parameter scheduler(s): {step_scheduler_names} and "
                f"{len(epoch_scheduler_names)} epoch-wise parameter scheduler(s): {epoch_scheduler_names}."
            )

    def get_step_scheduler_values(self) -> list[Any]:
        """Return step-wise scheduler values based on current global step."""
        if not hasattr(self, "global_step"):
            msg = (
                f"[{self.__class__.__name__}] global_step attribute not found in model."
            )
            logger.error(msg)
            raise ValueError(msg)

        return [
            scheduler.get_value(self.global_step)  # type: ignore
            for scheduler in self.other_schedulers_step
        ]

    def get_epoch_scheduler_values(self) -> list[Any]:
        """Return epoch-wise scheduler values based on current global epoch."""
        if not hasattr(self, "current_epoch"):
            msg = f"[{self.__class__.__name__}] current_epoch attribute not found in model."
            logger.error(msg)
            raise ValueError(msg)

        return [
            scheduler.get_value(self.current_epoch)  # type: ignore
            for scheduler in self.other_schedulers_epoch
        ]


@register(RK.PL_MODULE_MIXIN)
class OptimConfigMixin(PLModuleMixin):
    """Adds support for configuring optimizers and LR schedulers for Pytorch
    Lightning modules.

    Optimizer kwargs are expected to have the following format:
    ```
        optimizer_kwargs:
            {
            "name": "AdamW",
            "auto_no_weight_decay": true,
            "param_groups": [
                {
                "param_group_prefix": "encoder",
                "lr": 1e-3,
                "weight_decay": 1e-3
                }
            ]
            }

        # or single-group:
        optimizer_kwargs:
            {
            "name": "AdamW",
            "auto_no_weight_decay": true,
            "param_group_prefix": "encoder",
            "lr": 1e-3,
            "weight_decay": 1e-3
            }
    ```
    """

    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]]
    lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None

    def setup_mixin(
        self,
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Parse optimizer and LR scheduler configuration.

        Retrieves requested parameter groups from pl.LightningModule.model and
        parses them into optimizer_kwargs.

        Filters out parameters that should not receive weight decay--defined per optimizer:
        - If `auto_no_weight_decay` is True, weight decay is set to 0 for parameters that
        are not typically decayed, such as bias parameters, normalization parameters, etc.
        - If `no_weight_decay_prefixes` is provided, it is used to determine which parameters
        should not decay.
        """
        if not isinstance(optimizer_kwargs, (dict, list)):
            msg = (
                f"[{self.__class__.__name__}] `optimizer_kwargs` must be a dict, "
                f"list, or None, got {type(optimizer_kwargs).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(lr_scheduler_kwargs, (dict, list, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `lr_scheduler_kwargs` must be a dict, "
                f"list, or None, got {type(lr_scheduler_kwargs).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, "model"):
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'model' "
                "before calling OptimConfigMixin.setup_mixin()."
            )
            logger.error(msg)
            raise ValueError(msg)

        optimizer_kwargs = deepcopy(optimizer_kwargs)
        lr_scheduler_kwargs = (
            deepcopy(lr_scheduler_kwargs) if lr_scheduler_kwargs else None
        )

        if not isinstance(optimizer_kwargs, list):  # single optimizer
            optimizer_kwargs = [optimizer_kwargs]

        for cfg in optimizer_kwargs:
            if "param_groups" in cfg and "param_group_prefix" in cfg:
                logger.warning(
                    f"[{self.__class__.__name__}] `param_groups` and `param_group_prefix` "
                    "used together, `param_group_prefix` will be ignored."
                )
            # Weight decay settings
            auto_no_wd = bool(cfg.pop("auto_no_weight_decay", False))
            extra_nd_prefixes = cfg.pop("no_weight_decay_prefixes", None)
            if isinstance(extra_nd_prefixes, str):
                extra_nd_prefixes = [extra_nd_prefixes]

            resolved_groups: list[dict[str, Any]] = []
            no_wd_indices: list[int] = []
            if "param_groups" in cfg:
                # Multiple parameter groups inside this optimizer
                if not isinstance(cfg["param_groups"], list):
                    msg = (
                        f"[{self.__class__.__name__}] `param_groups` must be a list of dicts, "
                        f"got {type(cfg['param_groups']).__name__}."
                    )
                    logger.error(msg)
                    raise TypeError(msg)

                if "lr" in cfg or "weight_decay" in cfg:
                    logger.warning(
                        f"[{self.__class__.__name__}] `lr` and `weight_decay` are provided at the"
                        "top-level of the optimizer config; will get passed to subsequent sub-groups."
                    )

                for i, group_cfg in enumerate(cfg.pop("param_groups")):
                    named_params = cast(
                        list[tuple[str, nn.Parameter]],
                        get_parameter_groups_from_prefixes(
                            self.model,  # type: ignore[attr-defined]
                            group_cfg.pop("param_group_prefix", None),
                            return_named=True,
                        ),
                    )
                    wd_params, no_wd_params = split_decay_groups_from_params(
                        self.model,  # type: ignore[attr-defined]
                        named_params,
                        auto_no_weight_decay=auto_no_wd,
                        no_weight_decay_prefixes=extra_nd_prefixes,
                    )
                    # duplicate group: same hparams, but WD=0 for the no-decay subset
                    base = {
                        k: v
                        for k, v in group_cfg.items()
                        if k not in ("params", "name")
                    }  # keep lr, weight_decay, etc.
                    if wd_params:
                        resolved_groups.append({**base, "params": wd_params})
                    if no_wd_params:
                        resolved_groups.append(
                            {**base, "params": no_wd_params, "weight_decay": 0.0}
                        )
                        no_wd_indices.append(i)
            else:
                # Single parameter group
                prefix = cfg.pop("param_group_prefix", None)
                named_params = cast(
                    list[tuple[str, nn.Parameter]],
                    get_parameter_groups_from_prefixes(
                        self.model,  # type: ignore[attr-defined]
                        prefix,
                        return_named=True,
                    ),
                )
                wd_params, no_wd_params = split_decay_groups_from_params(
                    self.model,  # type: ignore[attr-defined]
                    named_params,
                    auto_no_weight_decay=auto_no_wd,
                    no_weight_decay_prefixes=extra_nd_prefixes,
                )
                base = {
                    k: v for k, v in cfg.items() if k not in ("params", "name")
                }  # entry contains optimizer name
                if wd_params:
                    resolved_groups.append({**base, "params": wd_params})
                if no_wd_params:
                    resolved_groups.append(
                        {**base, "params": no_wd_params, "weight_decay": 0.0}
                    )
                    no_wd_indices.append(0)

            cfg["params"] = resolved_groups

            # Track newly introduced param groups to duplicate scheduler settings
            def _dup_by_indices(vals: list, dup_idxs: list[int]) -> list:
                dup = set(dup_idxs or [])
                out = []
                for idx, v in enumerate(vals):
                    out.append(v)
                    if idx in dup:
                        out.append(v)  # duplicate this entr
                return out

            if isinstance(lr_scheduler_kwargs, list):
                # per-optimizer scheduler config
                for k, v in list(lr_scheduler_kwargs[i].items()):
                    if isinstance(v, list):
                        lr_scheduler_kwargs[i][k] = _dup_by_indices(v, no_wd_indices)
                    else:
                        lr_scheduler_kwargs[i][k] = v
            elif isinstance(lr_scheduler_kwargs, dict):
                # single scheduler dict
                for k, v in list(lr_scheduler_kwargs.items()):
                    if isinstance(v, list):
                        lr_scheduler_kwargs[k] = _dup_by_indices(v, no_wd_indices)
                    else:
                        lr_scheduler_kwargs[k] = v

        # Return to original format if only one optimizer
        self.optimizer_kwargs = (
            optimizer_kwargs if len(optimizer_kwargs) > 1 else optimizer_kwargs[0]
        )
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        logger.info(
            f"[{self.__class__.__name__}] Optimizer configuration:\n"
            f"{format_optimizer_log(self.optimizer_kwargs)}"
        )

    def get_optimizers_and_schedulers(
        self,
    ) -> (
        optim.Optimizer
        | dict[str, Any]
        | list[optim.Optimizer]
        | tuple[list[optim.Optimizer], list[dict[str, Any]]]
    ):
        """Get optimizers and schedulers.

        This method is to be called in configure_optimizers of the module.

        Aligning with Lighnting's documentation, it can return the following:
        (depending on the optimizer_kwargs and scheduler_kwargs inputs)
        - A single optimizer
        - A dict with a single optimizer and a single lr_scheduler_config
        - A list of optimizers
        - A list of optimizers with a list of lr_scheduler_configs
        """
        if not hasattr(self, "optimizer_kwargs"):
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'optimizer_kwargs' "
                "before calling OptimConfigMixin.get_optimizers_and_schedulers()."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "lr_scheduler_kwargs"):
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'lr_scheduler_kwargs' "
                "before calling OptimConfigMixin.get_optimizers_and_schedulers()."
            )
            logger.error(msg)
            raise ValueError(msg)

        optimizer = UnitFactory.get_optimizer_instances_from_kwargs(
            self.optimizer_kwargs
        )

        if self.lr_scheduler_kwargs is not None:
            lr_scheduler = UnitFactory.get_lr_scheduler_instances_from_kwargs(
                self.lr_scheduler_kwargs, optimizer
            )
        else:
            lr_scheduler = None

        if isinstance(optimizer, list) and lr_scheduler is not None:
            if len(optimizer) != len(lr_scheduler):
                msg = "Number of optimizers and LR schedulers must match."
                logger.error(msg)
                raise ValueError(msg)
            return optimizer, cast(list[dict[str, Any]], lr_scheduler)

        if isinstance(optimizer, list) and lr_scheduler is None:
            return optimizer

        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

        return optimizer


@register(RK.PL_MODULE_MIXIN)
class InfererMixin(PLModuleMixin):
    """Adds support for model inference."""

    inferer: Inferer
    postprocess: Callable | Compose | Transform
    tta: list[Compose] | None

    def setup_mixin(
        self,
        inference_settings: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """Create inferer instance using the dedicated factory.

        If `postprocess` is provided, it is resolved as a `Compose` object.
            Expected to be a single--or a list of--array-based transforms.
        If `tta` is provided, it is resolved as a list of `Compose` objects.
            Each element is expected to comprise dictionary transforms,
            e.g., `Flipd`, `RandAffined`, etc, which are needed for inverting
            the model outputs.
        If `inferer_kwargs` is provided, it is used to instantiate the inferer.
        """
        if not isinstance(inference_settings, (dict, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `inference_settings` must be a dict, "
                f"got {type(inference_settings).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        if hasattr(self, "inferer"):
            logger.warning(
                f"[{self.__class__.__name__}] Attribute `inferer' already exists "
                "and will be overwritten."
            )

        inference_settings = inference_settings or {}
        inference_settings = deepcopy(inference_settings)

        # Resolve postprocess, tta, and inferer
        postprocess = resolve_transform(inference_settings.get("postprocess"))
        if isinstance(postprocess, list):
            self.postprocess = Compose(postprocess)
        elif callable(postprocess):
            self.postprocess = postprocess
        elif postprocess is None:
            self.postprocess = Identity()
        else:
            msg = (
                f"[{self.__class__.__name__}] `postprocess` transforms must be a list of "
                f"transforms or a callable object, or None, got {type(postprocess).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        tta = resolve_transform(inference_settings.get("tta"))
        if isinstance(tta, list):
            self.tta = cast(list[Compose], tta)
        elif callable(tta):
            self.tta = [tta]
        elif tta is None:
            self.tta = None
        else:
            msg = (
                f"[{self.__class__.__name__}] `tta` must be a list of transforms, "
                f"a callable object, or None, got {type(tta).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        self.inferer = UnitFactory.get_inferer_instance_from_kwargs(
            inference_settings.get("inferer_kwargs", {"name": "SimpleInferer"})
        )

        logger.info(
            f"[InfererMixin] Instantiated inferer: {self.inferer.__class__.__name__}, "
            f"postprocess transforms: {callable_name(postprocess)}, "
            f"TTA transforms: {callable_name(tta)}."
        )

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, Any],
        img_key: str = "img",
        *,
        seed: int | None = None,
        return_std: bool = False,
        do_tta: bool = True,
        do_postprocess: bool = True,
        invert: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prediction pipeline for input tensor.

        Defaults to SimpleInferer, no postprocess, and no TTA.

        If TTA is enabled, the input tensor is decollated, processed through each TTA transform,
        the model is applied (collated back), and the output is inverted (decollate->collate) if
        the TTA transform is invertible; i.e., must implement the `InvertibleTransform` interface,
        which is true for any MONAI `Compose` object. The raw logits are then averaged and
        postprocessed.

        If `return_std` is True, the standard deviation of the predictions is returned.
        If `seed` is provided, the predictions are deterministic.
        If `do_tta` is False, TTA is disabled; defaults to True.
        """
        if not hasattr(self, "model"):
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'model' "
                "before calling InfererMixin.predict()."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not self.tta or not do_tta:
            output = self.inferer(batch[img_key], self.model)  # type: ignore[attr-defined]
            if do_postprocess:
                output = self.postprocess(output)
            if return_std:
                logger.warning(
                    "TTA disabled/empty; cannot compute std. Returning only mean."
                )
            return cast(torch.Tensor, output)

        # TTA loop
        mean: torch.Tensor | None = None
        m2: torch.Tensor | None = None
        n = 0
        for i, tta_run in enumerate(self.tta):
            # Apply TTA to batch
            aug_batch, invert_fn = apply_tta_monai_batch(
                batch,
                tta_run,
                img_key=img_key,
                seed=(None if seed is None else seed + i),
            )
            # Forward inference pass on augmented batch
            aug_batch["logits"] = self.inferer(aug_batch[img_key], self.model)  # type: ignore[attr-defined]

            # Invert TTA on logits
            if invert:
                out_logits = invert_fn(
                    aug_batch,
                    out_key="logits",
                    nearest_interp=False,
                    device=batch[img_key].device,
                )["logits"]
            else:
                out_logits = aug_batch["logits"]

            # Online mean/variance update
            n += 1
            if mean is None:
                mean = out_logits.clone()
                if return_std:
                    m2 = torch.zeros_like(out_logits)
            else:
                delta = out_logits - mean
                mean = mean + delta / n
                if return_std:
                    m2 = m2 + delta * (out_logits - mean)

            del aug_batch, out_logits

        # Finalize
        output = cast(torch.Tensor, mean)
        if do_postprocess:
            output = cast(torch.Tensor, self.postprocess(output))

        if return_std:
            m2 = cast(torch.Tensor, m2)
            var = m2 / (n - 1) if n > 1 else torch.zeros_like(output)
            std = torch.sqrt(torch.clamp(var, min=0))
            return output, std

        return output


@register(RK.PL_MODULE_MIXIN)
class HParamsMixin(PLModuleMixin):
    """Adds support for saving and logging hyperparameters."""

    def setup_mixin(
        self,
        ignore_hparams: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Save and log hyperparameters."""
        if not isinstance(ignore_hparams, (list, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `ignore_hparams` must be a list or None, "
                f"got {type(ignore_hparams).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        pl_module = cast(plModule, self)

        if ignore_hparams is None:
            ignore_hparams = []

        pl_module.save_hyperparameters(
            ignore=["ignore_hparams"] + ignore_hparams, logger=True
        )

        logger.info(
            f"[{self.__class__.__name__}] Lightning module initialized with following "
            f"hyperparameters (`pl_module.hparams`): \n{pl_module.hparams}"
        )


@register(RK.PL_MODULE_MIXIN)
class ArtifactsMixin(PLModuleMixin):
    """Adds support for saving and loading artifacts."""

    def setup_mixin(
        self,
        artifacts_settings: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """Configure logging of artifacts (now supporting only tensors) at
        regular intervals."""
        if not isinstance(artifacts_settings, (dict, type(None))):
            msg = (
                f"[{self.__class__.__name__}] `artifacts_settings` must be a dict, "
                f"got {type(artifacts_settings).__name__}."
            )
            logger.error(msg)
            raise TypeError(msg)

        artifacts_settings = artifacts_settings or {}
        log_every_n_steps = artifacts_settings.get("log_every_n_steps", 0)
        log_max_n_items = artifacts_settings.get("log_max_n_items", 1)

        if (
            not isinstance(log_every_n_steps, int)
            or not isinstance(log_max_n_items, int)
            or log_every_n_steps < 0
            or log_max_n_items < 0
        ):
            msg = f"[{self.__class__.__name__}] `log_every_n_steps` and `log_max_n_items` must be positive integers."
            logger.error(msg)
            raise ValueError(msg)

        self.log_every_n_steps = log_every_n_steps
        self.log_max_n_items = log_max_n_items

    @rank_zero_only
    @torch.no_grad()
    def log_tensors_dict(self, tensors_dict: dict[str, torch.Tensor]) -> None:
        """Log a dictionary of tensors."""
        current_step = getattr(self, "global_step", None)
        if current_step is None:
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'global_step' "
                "before calling ArtifactsMixin.log_tensors_dict()."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "log_every_n_steps") or not hasattr(
            self, "log_max_n_items"
        ):
            msg = (
                f"[{self.__class__.__name__}] Expected attributes 'log_every_n_steps' and 'log_max_n_items' "
                "before calling ArtifactsMixin.log_tensors_dict()."
            )
            logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "logger"):
            msg = (
                f"[{self.__class__.__name__}] Expected attribute 'logger' "
                "before calling ArtifactsMixin.log_tensors_dict()."
            )
            logger.error(msg)
            raise ValueError(msg)

        if self.logger is None:  # type: ignore[attr-defined]
            return

        current_step = int(current_step)
        if (
            not self.log_every_n_steps
            or not self.log_max_n_items
            or current_step % self.log_every_n_steps != 0
        ):
            return

        if not isinstance(tensors_dict, dict) or not all(
            isinstance(v, torch.Tensor) for v in tensors_dict.values()
        ):
            msg = f"[{self.__class__.__name__}] `tensors_dict` must be a dictionary of tensors."
            logger.error(msg)
            raise TypeError(msg)

        log_dict: dict[str, torch.Tensor] = {}
        for k, v in tensors_dict.items():
            # Limit items
            n = min(v.shape[0], self.log_max_n_items)
            v = v[:n]

            # Get mid slice of last dimension if 3D
            if v.ndim - 2 == 3:  # 3D
                img_slice = v[..., v.shape[-1] // 2].detach().float().cpu()
            elif v.ndim - 2 == 2:  # 2D
                img_slice = v.detach().float().cpu()
            else:
                msg = (
                    f"[{self.__class__.__name__}] Expected tensor (B, C, *spatial_dims) with 2 or 3 "
                    f"spatial dimensions; got {v.ndim - 2}."
                )
                logger.error(msg)
                raise ValueError(msg)

            # Convert to loggable format: get unique 2D images for
            # each batch item-channel pair (for the first `n` batch items).
            for i in range(n):
                for ch in range(v.shape[1]):
                    log_dict[f"{k}_idx{i}_ch{ch}"] = img_slice[i, ch]

        if self.logger.__class__.__name__.lower().startswith("wandb"):  # type: ignore[attr-defined]
            import wandb

            self.logger.experiment.log(  # type: ignore[attr-defined]
                {
                    k: wandb.Image(
                        to_uint8_image(v, clamp_perc=(1.0, 99.0))
                    )
                    for k, v in log_dict.items()
                },
                step=current_step,
            )
            return

        if self.logger.__class__.__name__.lower().startswith("tensorboard"):  # type: ignore[attr-defined]
            for k, v in log_dict.items():
                self.logger.experiment.add_image(  # type: ignore[attr-defined]
                    k,
                    to_uint8_image(
                        v, clamp_perc=(1.0, 99.0)
                    ).unsqueeze(0),
                    global_step=current_step,
                )
            return
