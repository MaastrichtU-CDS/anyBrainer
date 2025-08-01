"""Mixin classes for Pytorch Lightning modules."""

from __future__ import annotations

__all__ = [
    "ModelInitMixin",
    "LossMixin",
    "ParamSchedulerMixin",
    "OptimConfigMixin",
    "WeightInitMixin",
    "HParamsMixin",
]

import logging
from typing import Any, Callable, TYPE_CHECKING, cast
from copy import deepcopy

from lightning.pytorch import LightningModule as plModule

from anyBrainer.core.utils import (
    get_parameter_groups_from_prefixes,
    load_param_group_from_ckpt,
    resolve_path,
)
from anyBrainer.core.engines.utils import (
    resolve_fn,
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
    """
    Adds support for instantiating the model.
    """
    model: nnModule
    
    def setup_mixin(
        self, 
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> None:
        """Create model instance using the dedicated factory."""
        if not isinstance(model_kwargs, dict):
            msg = (f"[{self.__class__.__name__}] `model_kwargs` must be a dict, "
                   f"got {type(model_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)

        if hasattr(self, "model"):
            logger.warning(f"[{self.__class__.__name__}] Attribute `model' already exists "
                           "and will be overwritten.")
        
        self.model = UnitFactory.get_model_instance_from_kwargs(model_kwargs)


@register(RK.PL_MODULE_MIXIN)
class LossMixin(PLModuleMixin):
    """
    Adds support for instantiating the loss function(s).
    """
    loss_fn: nnModule | list[nnModule]
    
    def setup_mixin(
        self,
        loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
        **kwargs,
    ) -> None:
        """Create single or list of loss fn instances using the dedicated factory."""
        if not isinstance(loss_fn_kwargs, (list, dict)):
            msg = (f"[{self.__class__.__name__}] `loss_fn_kwargs` must be a dict, "
                   f"or list, got {type(loss_fn_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)

        if hasattr(self, "loss_fn"):
            logger.warning(f"[{self.__class__.__name__}] Attribute `loss_fn' already exists "
                           "and will be overwritten.")
        
        self.loss_fn = UnitFactory.get_loss_fn_instances_from_kwargs(loss_fn_kwargs)


@register(RK.PL_MODULE_MIXIN)
class ParamSchedulerMixin(PLModuleMixin):
    """
    Adds support for extra (non-optimizer) schedulers; e.g., contrastive 
    learning momentum, loss function weighting, etc.
    """
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
            msg = (f"[{self.__class__.__name__}] `param_scheduler_kwargs` must be a list or None, "
                   f"got {type(param_scheduler_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)
    
        if param_scheduler_kwargs is not None:
            self.other_schedulers_step, self.other_schedulers_epoch = (
                UnitFactory.get_param_scheduler_instances_from_kwargs(param_scheduler_kwargs)
            )

    def get_step_scheduler_values(self) -> list[Any]:
        """Return step-wise scheduler values based on current global step."""
        if not hasattr(self, "global_step"):
            msg = (f"[{self.__class__.__name__}] global_step attribute not found in model.")
            logger.error(msg)
            raise ValueError(msg)

        return [
            scheduler.get_value(self.global_step) # type: ignore
            for scheduler in self.other_schedulers_step
        ]

    def get_epoch_scheduler_values(self) -> list[Any]:
        """Return epoch-wise scheduler values based on current global epoch."""
        if not hasattr(self, "current_epoch"):
            msg = (f"[{self.__class__.__name__}] current_epoch attribute not found in model.")
            logger.error(msg)
            raise ValueError(msg)

        return [
            scheduler.get_value(self.current_epoch) # type: ignore
            for scheduler in self.other_schedulers_epoch
        ]


@register(RK.PL_MODULE_MIXIN)
class OptimConfigMixin(PLModuleMixin):
    """
    Adds support for configuring optimizers and LR schedulers for 
    Pytorch Lightning modules.
    """
    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]]
    lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None
    
    def setup_mixin(
        self,
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        lr_scheduler_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """
        Parse optimizer and LR scheduler configuration.
        
        Retrieves requested parameter groups from pl.LightningModule.model and 
        parses them into optimizer_kwargs. 
        """
        if not isinstance(optimizer_kwargs, (dict, list)):
            msg = (f"[{self.__class__.__name__}] `optimizer_kwargs` must be a dict, "
                   f"list, or None, got {type(optimizer_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)
        
        if not isinstance(lr_scheduler_kwargs, (dict, list, type(None))):
            msg = (f"[{self.__class__.__name__}] `lr_scheduler_kwargs` must be a dict, "
                   f"list, or None, got {type(lr_scheduler_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)    

        optimizer_kwargs = deepcopy(optimizer_kwargs)
        lr_scheduler_kwargs = deepcopy(lr_scheduler_kwargs) if lr_scheduler_kwargs else None

        if not hasattr(self, "model"):
            msg = (f"[{self.__class__.__name__}] Expected attribute 'model' "
                   "before calling OptimConfigMixin.setup_mixin().")
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(optimizer_kwargs, list):
            for cfg in optimizer_kwargs:
                cfg["param_groups"] = get_parameter_groups_from_prefixes(
                    self.model, cfg.pop("param_group_prefix", None) # type: ignore
                )
        else:
            optimizer_kwargs["param_groups"] = get_parameter_groups_from_prefixes(
                self.model, optimizer_kwargs.pop("param_group_prefix", None) # type: ignore
            )

        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
    
    def get_optimizers_and_schedulers(
        self,
    ) -> (optim.Optimizer | dict[str, Any] | list[optim.Optimizer] | 
          tuple[list[optim.Optimizer], list[dict[str, Any]]]):
        """
        Get optimizers and schedulers.
        
        This method is to be called in configure_optimizers of the module.
        
        Aligning with Lighnting's documentation, it can return the following:
        (depending on the optimizer_kwargs and scheduler_kwargs inputs)
        - A single optimizer
        - A dict with a single optimizer and a single lr_scheduler_config
        - A list of optimizers
        - A list of optimizers with a list of lr_scheduler_configs
        """
        if not hasattr(self, "optimizer_kwargs"):
            msg = (f"[{self.__class__.__name__}] Expected attribute 'optimizer_kwargs' "
                   "before calling OptimConfigMixin.get_optimizers_and_schedulers().")
            logger.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "lr_scheduler_kwargs"):
            msg = (f"[{self.__class__.__name__}] Expected attribute 'lr_scheduler_kwargs' "
                   "before calling OptimConfigMixin.get_optimizers_and_schedulers().")
            logger.error(msg)
            raise ValueError(msg)

        optimizer = (
            UnitFactory.get_optimizer_instances_from_kwargs(self.optimizer_kwargs)
        )

        if self.lr_scheduler_kwargs is not None:
            lr_scheduler = (
                UnitFactory.get_lr_scheduler_instances_from_kwargs(self.lr_scheduler_kwargs, optimizer)
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
class WeightInitMixin(PLModuleMixin):
    """
    Adds support for initializing model weights.
    """
    def setup_mixin(
        self,
        weights_init_kwargs: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """
        Initialize model weights either through a pretrained checkpoint or a 
        custom weight initialization function.
        """
        if not isinstance(weights_init_kwargs, (dict, type(None))):
            msg = (f"[{self.__class__.__name__}] `weights_init_kwargs` must be a dict, "
                   f"or None, got {type(weights_init_kwargs).__name__}.")
            logger.error(msg)
            raise TypeError(msg)

        if not hasattr(self, "model"):
            msg = "Expected attribute 'model' before calling WeightInitMixin.setup_mixin()."
            logger.error(msg)
            raise ValueError(msg)
        
        if weights_init_kwargs is None:
            weights_init_kwargs = {}
        
        weights_init_fn = weights_init_kwargs.get("weights_init_fn")
        load_pretrain_weights = weights_init_kwargs.get("load_pretrain_weights")
        load_param_group_prefix = weights_init_kwargs.get("load_param_group_prefix")
        extra_load_kwargs = weights_init_kwargs.get("extra_load_kwargs", {})

        if load_pretrain_weights is not None:
            if weights_init_fn is not None:
                logger.warning(f"[{self.__class__.__name__}] Provided both "
                                f"load_pretrain_weights and weights_init_fn. "
                                f"weights_init_fn will be ignored.")
            self._load_pretrain_weights(
                load_pretrain_weights=load_pretrain_weights,
                load_param_group_prefix=load_param_group_prefix,
                extra_load_kwargs=extra_load_kwargs,
            )
            return

        if weights_init_fn is not None:
            self._apply_weights_init_fn(
                weights_init_fn=weights_init_fn,
            )
            return

        logger.info(f"[{self.__class__.__name__}] Model initialized with random weights.")
    
    def _load_pretrain_weights(
        self,
        load_pretrain_weights: str,
        load_param_group_prefix: str | list[str] | None,
        extra_load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Load pretrained weights from checkpoint.
        
        Allows selective loading of specific parameter groups. 
        """
        try:
            self.model, stats = load_param_group_from_ckpt(
                model_instance=self.model,
                checkpoint_path=resolve_path(load_pretrain_weights),
                param_group_prefix=load_param_group_prefix,
                extra_load_kwargs=extra_load_kwargs,
            )
            logger.info(f"[{self.__class__.__name__}] Loaded pretrained weights from checkpoint "
                        f"{load_pretrain_weights}\n#### Summary ####"
                        f"\n- Attempted to load {len(stats['loaded_keys'])} parameters "
                        f"(filtered out {len(stats['ignored_keys'])} parameters)"
                            f"\n- Unexpected {len(stats['unexpected_keys'])} parameters"
                            f"\n- Missing {len(stats['missing_keys'])} parameters")
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Failed to load pretrained weights: {e}")
            raise e
    
    def _apply_weights_init_fn(
        self,
        weights_init_fn: Callable | str,
    ) -> None:
        """Apply weight initialization function to model."""
        try:
            self.model.apply(cast(Callable, resolve_fn(weights_init_fn)))
            logger.info(f"[{self.__class__.__name__}] Initialized model weights with "
                        f"{weights_init_fn}.")
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Failed to apply weight initialization function: {e}")
            raise e
    

@register(RK.PL_MODULE_MIXIN)
class HParamsMixin(PLModuleMixin):
    """
    Adds support for saving and logging hyperparameters.
    """
    def setup_mixin(
        self,
        ignore_hparams: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Save and log hyperparameters."""
        if not isinstance(ignore_hparams, (list, type(None))):
            msg = (f"[{self.__class__.__name__}] `ignore_hparams` must be a list or None, "
                   f"got {type(ignore_hparams).__name__}.")
            logger.error(msg)
            raise TypeError(msg)

        pl_module = cast(plModule, self)

        if ignore_hparams is None:
            ignore_hparams = []

        pl_module.save_hyperparameters(
            ignore=["ignore_hparams"] + ignore_hparams, logger=True
        )

        logger.info(f"[{self.__class__.__name__}] Lightning module initialized with following "
                    f"hyperparameters:\n{pl_module.hparams}")