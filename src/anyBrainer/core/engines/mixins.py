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

import torch
from lightning.pytorch import LightningModule as plModule
from monai.inferers.inferer import Inferer
from monai.transforms.utility.array import Identity
#pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    Invertd,
    InvertibleTransform,
    Randomizable,
    EnsureType,
    Transform,
)

from anyBrainer.core.utils import (
    get_parameter_groups_from_prefixes,
    load_param_group_from_ckpt,
    resolve_path,
    callable_name,
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

        logger.info(f"[{self.__class__.__name__}] Instantiated model: {model_kwargs['name']}.")


@register(RK.PL_MODULE_MIXIN)
class WeightInitMixin(PLModuleMixin):
    """
    Adds support for initializing model weights.
    """
    def setup_mixin(
        self,
        weights_init_settings: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """
        Initialize model weights either through a pretrained checkpoint or a 
        custom weight initialization function.
        """
        if not isinstance(weights_init_settings, (dict, type(None))):
            msg = (f"[{self.__class__.__name__}] `weights_init_settings` must be a dict, "
                   f"or None, got {type(weights_init_settings).__name__}.")
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
            logger.info(f"[{self.__class__.__name__}] Model initialized with random weights.")
            return

        if weights_init_fn is not None:
            self._apply_weights_init_fn(
                weights_init_fn=weights_init_fn,
            )
        if load_pretrain_weights is not None:
            if weights_init_fn is not None:
                logger.warning(f"[{self.__class__.__name__}] Provided both "
                                f"`weights_init_fn` and `load_pretrain_weights`; "
                                f"the latter can override weights initialized by "
                                f"`{weights_init_fn}`.")
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
        strict: bool = True,
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
                select_prefixes=load_param_group_prefix,
                rename_map=rename_map,
                strict=strict,
                torch_load_kwargs=extra_load_kwargs,
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
                        f"`{weights_init_fn}`.")
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Failed to apply weight initialization function: {e}")
            raise e


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

        if isinstance(self.loss_fn, list):
            loss_fn_names = [loss_fn.__class__.__name__ for loss_fn in self.loss_fn]
            logger.info(f"[{self.__class__.__name__}] Instantiated loss function(s): {loss_fn_names}.")
        else:
            logger.info(f"[{self.__class__.__name__}] Instantiated loss function: {self.loss_fn.__class__.__name__}.")


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
            step_scheduler_names = [scheduler.__class__.__name__ for scheduler in self.other_schedulers_step]
            epoch_scheduler_names = [scheduler.__class__.__name__ for scheduler in self.other_schedulers_epoch]
            
            logger.info(
                f"[{self.__class__.__name__}] Instantiated {len(step_scheduler_names)} "
                f"step-wise parameter scheduler(s): {step_scheduler_names} and "
                f"{len(epoch_scheduler_names)} epoch-wise parameter scheduler(s): {epoch_scheduler_names}."
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

        if not isinstance(optimizer_kwargs, list):
            optimizer_kwargs = [optimizer_kwargs]

        for cfg in optimizer_kwargs:
            if "param_groups" in cfg and "param_group_prefix" in cfg:
                logger.warning(f"[{self.__class__.__name__}] `param_groups` and `param_group_prefix` "
                               "used together, `param_group_prefix` will be ignored.")
            
            if "param_groups" in cfg:
                # Multiple parameter groups inside this optimizer
                if not isinstance(cfg["param_groups"], list):
                    msg = (f"[{self.__class__.__name__}] `param_groups` must be a list of dicts, "
                           f"got {type(cfg['param_groups']).__name__}.")
                    logger.error(msg)
                    raise TypeError(msg)

                resolved_groups = []
                for group_cfg in cfg.pop("param_groups"):
                    group_cfg["params"] = get_parameter_groups_from_prefixes(
                        self.model, group_cfg.pop("param_group_prefix", None) # type: ignore[attr-defined]
                    )
                    resolved_groups.append(group_cfg)
                cfg["params"] = resolved_groups
            else:
                # Single parameter group
                prefix = cfg.pop("param_group_prefix", None)
                cfg["params"] = get_parameter_groups_from_prefixes(self.model, prefix) # type: ignore[attr-defined]

        # Return to original format if only one optimizer
        self.optimizer_kwargs = (optimizer_kwargs if len(optimizer_kwargs) > 1 
                                 else optimizer_kwargs[0])
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        logger.info(f"[{self.__class__.__name__}] Optimizer configuration:\n"
                    f"{format_optimizer_log(self.optimizer_kwargs)}")

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
class InfererMixin(PLModuleMixin):
    """
    Adds support for model inference.
    """
    inferer: Inferer
    postprocess: Callable | Compose | Transform
    tta: list[Callable | Compose | Transform] | None

    def setup_mixin(
        self,
        inference_settings: dict[str, Any] | None,
        **kwargs,
    ) -> None:
        """Create inferer instance using the dedicated factory."""
        if not isinstance(inference_settings, (dict, type(None))):
            msg = (f"[{self.__class__.__name__}] `inference_settings` must be a dict, "
                   f"got {type(inference_settings).__name__}.")
            logger.error(msg)
            raise TypeError(msg)
        
        if hasattr(self, "inferer"):
            logger.warning(f"[{self.__class__.__name__}] Attribute `inferer' already exists "
                           "and will be overwritten.")

        inference_settings = inference_settings or {}
        inference_settings = deepcopy(inference_settings)

        # Resolve postprocess, tta, and inferer
        self.postprocess = Compose(resolve_transform(inference_settings.get("postprocess")))
        self.tta = resolve_transform(inference_settings.get("tta"))
        self.inferer = UnitFactory.get_inferer_instance_from_kwargs(
            inference_settings.get("inferer_kwargs", {"name": "SimpleInferer"})
        )

        logger.info(f"[InfererMixin] Instantiated inferer: {self.inferer.__class__.__name__}, "
                    f"postprocess transforms: {callable_name(self.postprocess)}.")

    @torch.no_grad()
    def predict(
        self, 
        input: torch.Tensor,
        *,
        seed: int | None = None,
        return_std: bool = False,
        do_tta: bool = True,
        do_postprocess: bool = True,
        clone_tensor: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction pipeline for input tensor.

        Defaults to SimpleInferer, Identity() postprocess, and no TTA. 
        
        If TTA is enabled, the input tensor is processed through each TTA transform,
        the model is applied, and the output is inverted if the TTA transform is invertible;
        i.e., must implement MONAI's `InvertibleTransform` interface, which is true for any 
        `Compose` object. The raw logits are then averaged and postprocessed. 
        
        If `return_std` is True, the standard deviation of the predictions is returned.
        If `seed` is provided, the predictions are deterministic.
        If `do_tta` is False, TTA is disabled; defaults to True.
        If `clone_tensor` is True, the input tensor is cloned before processing; 
        defaults to False for max efficiency, since typical MONAI and PyTorch operations 
        do not modify tensors in place.
        """
        if clone_tensor:
            input = input.detach().clone()

        if not hasattr(self, "model"):
            msg = (f"[{self.__class__.__name__}] Expected attribute 'model' "
                   "before calling InfererMixin.predict().")
            logger.error(msg)
            raise ValueError(msg)

        if not self.tta or not do_tta:
            output = self.inferer(input, self.model) # type: ignore[attr-defined]
            if do_postprocess:
                output = self.postprocess(output)
            return cast(torch.Tensor, output)

        # TTA loop
        mean = None
        m2 = None
        n = 0
        ensure_meta = EnsureType(track_meta=True)
        for i, tta_run in enumerate(self.tta):
            # Optional determinism for randomizable transforms
            try:
                if seed is not None and hasattr(tta_run, "set_random_state"):
                    tta_run.set_random_state(seed=seed + i) # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] set_random_state failed: {e}")
            
            current_in = tta_run(ensure_meta(input))
            current_logits = self.inferer(current_in, self.model) # type: ignore[attr-defined]

            inv_logits = current_logits
            if isinstance(tta_run, InvertibleTransform):
                invert_op = Invertd(keys="logits", transform=tta_run, orig_keys="input")
                inv_logits = invert_op({"input": current_in, "logits": current_logits})["logits"]
            else:
                logger.warning(f"[{self.__class__.__name__}] TTA transform is not "
                                "invertible; accumulating without inversion.")
            # Online mean/variance update
            n += 1
            if mean is None:
                mean = inv_logits.clone()
                if return_std:
                    m2 = torch.zeros_like(inv_logits)
            else:
                delta = inv_logits - mean
                mean = mean + delta / n
                if return_std:
                    m2 = m2 + delta * (inv_logits - mean)

        # Finalize
        output = mean
        if do_postprocess:
            output = self.postprocess(output)

        if return_std:
            var = m2 / (n - 1) if n > 1 else torch.zeros_like(mean) # type: ignore
            std = torch.sqrt(torch.clamp(var, min=0))
            return cast(torch.Tensor, output), cast(torch.Tensor, std)

        return cast(torch.Tensor, output)


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
                    f"hyperparameters (`pl_module.hparams`): \n{pl_module.hparams}")