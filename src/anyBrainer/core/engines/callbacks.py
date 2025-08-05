"""Define custom callbacks for Pytorch Lightning."""

from __future__ import annotations

__all__ = [
    "UpdateDatamoduleEpoch",
    "LogLR",
    "LogGradNorm",
    "FreezeParamGroups",
]

import logging
from typing import Any, TYPE_CHECKING

import lightning.pytorch as pl
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm

from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.utils import (
    get_optimizer_lr,
    get_total_grad_norm,
    get_parameter_groups_from_prefixes,
)
from anyBrainer.core.engines.utils import (
    sync_dist_safe,
)

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


@register(RK.CALLBACK)
class UpdateDatamoduleEpoch(pl.Callback):
    """
    Callback to update the datamodule's _current_epoch attribute.
    """        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sync the current epoch between the trainer and the datamodule."""
        if (trainer.datamodule is None or not hasattr(trainer.datamodule, "_current_epoch")): # pyright: ignore
            logger.warning("Datamodule does not have _current_epoch attribute. "
                           "This is likely due to a custom datamodule that does not "
                           "inherit from anyBrainer.data.BaseDataModule.")
            return
        
        trainer.datamodule._current_epoch = trainer.current_epoch + 1 # type: ignore
        logger.debug(f"Updated datamodule epoch to {trainer.current_epoch + 1}")


@register(RK.CALLBACK)
class LogLR(pl.Callback):
    """
    Callback to log the learning rate for all optimizers in the trainer.
    """
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        """Log the learning rate."""
        if not isinstance(trainer.optimizers, list):
            logger.warning("Cannot log LR because pl_module.optimizers is not a list.")
            return
        
        pl_module.log_dict(
            get_optimizer_lr(trainer.optimizers),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=sync_dist_safe(pl_module),
        )


@register(RK.CALLBACK)
class LogGradNorm(pl.Callback):
    """
    Callback to log the gradient norm for all parameters in the model.
    """
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log the gradient norm."""
        pl_module.log(
            "train/grad_norm",
            get_total_grad_norm(pl_module),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=sync_dist_safe(pl_module),
        )


@register(RK.CALLBACK)
class FreezeParamGroups(pl.Callback):
    """
    Freeze / unfreeze selected parameter groups at scheduled epochs.

    Args:
    - param_group_prefix: List of parameter group prefixes to freeze
    - freeze_epoch: Epoch at which to freeze the parameter groups;
        can be a single epoch or a list (len(freeze_epoch) == len(param_group_prefix)).
    - unfreeze_epoch: Epoch at which to unfreeze the parameter groups;
        same format as freeze_epoch.
    - train_bn: Whether to keep BatchNorm layers in train mode.
    """
    def __init__(
        self,
        *,
        param_group_prefix: str | list[str],
        freeze_epoch: int | list[int] = 0,
        unfreeze_epoch: int | list[int] = 0,
        train_bn: bool = False,
    ) -> None:
        super().__init__()
        # normalise prefixes
        if isinstance(param_group_prefix, str):
            param_group_prefix = [param_group_prefix]
        self.prefixes = param_group_prefix

        # normalise epochs
        def _expand(x: int | list[int]) -> list[int]:
            return [x] * len(self.prefixes) if isinstance(x, int) else x

        self.freeze_epoch = _expand(freeze_epoch)
        self.unfreeze_epoch = _expand(unfreeze_epoch)
        if len(self.freeze_epoch) != len(self.prefixes):
            msg = (f"[FreezeParamGroups] `freeze_epoch` must match number of parameter "
                   f"group prefixes, got {len(self.freeze_epoch)} and {len(self.prefixes)}.")
            logger.error(msg)
            raise ValueError(msg)
        if len(self.unfreeze_epoch) != len(self.prefixes):
            msg = (f"[FreezeParamGroups] `unfreeze_epoch` must match number of parameter "
                   f"group prefixes, got {len(self.unfreeze_epoch)} and {len(self.prefixes)}.")
            logger.error(msg)
            raise ValueError(msg)

        self.train_bn = train_bn
        self._are_frozen: list[bool] | None = None     # set in setup()
        self._param_groups: list[list[nn.Parameter]] | None = None  # set in setup()
    
    def setup(self, trainer, pl_module, stage: str | None = None):
        self._param_groups = [
            get_parameter_groups_from_prefixes(
                model=pl_module.model, # type: ignore[attr-defined]
                prefixes=pref,
                trainable_only=False,
                silent=True,
            )
            for pref in self.prefixes
        ]
        self._are_frozen = [False] * len(self._param_groups)

        logger.info(
            f"[FreezeParamGroups] managing {len(self._param_groups)} groups "
            f"(freeze at {self.freeze_epoch}, unfreeze at {self.unfreeze_epoch}, "
            f"train_bn={self.train_bn})."
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Freeze parameter groups."""
        epoch = trainer.current_epoch
        frozen, unfrozen = self._toggle_requires_grad(epoch)
        bn_mode = self._set_batchnorm_mode(pl_module)

        if frozen or unfrozen:
            msg = (f"[FreezeCallback] epoch {epoch} | "
                   f"froze: {frozen or '—'} | unfroze: {unfrozen or '—'}")
            if bn_mode is not None:
                msg += f" | BatchNorm train={bn_mode}"
            logger.info(msg)
    
    def _toggle_requires_grad(self, epoch: int) -> tuple[list[str], list[str]]:
        """
        Freeze/unfreeze groups, return lists of prefix names that changed.

        If same epoch is in both freeze and unfreeze lists, the parameter group 
        will be unfrozen.
        """
        if self._param_groups is None or self._are_frozen is None:
            msg = (f"[FreezeParamGroups] `_param_groups` or `_are_frozen` is None. "
                   "This is likely due to using the callback outside the trainer, "
                   "which automatically calls `setup()`.")
            logger.error(msg)
            raise ValueError(msg)
        
        frozen, unfrozen = [], []
        for i, params in enumerate(self._param_groups):
            if epoch == self.freeze_epoch[i] and not self._are_frozen[i]:
                for p in params:
                    p.requires_grad = False
                self._are_frozen[i] = True
                frozen.append(self.prefixes[i])

            if epoch == self.unfreeze_epoch[i] and self._are_frozen[i]:
                for p in params:
                    p.requires_grad = True
                self._are_frozen[i] = False
                unfrozen.append(self.prefixes[i])

        return frozen, unfrozen
    
    def _set_batchnorm_mode(self, pl_module) -> bool | None:
        """Put BN layers in train/eval depending on frozen status."""
        if self._are_frozen is None:
            msg = (f"[FreezeParamGroups] `_are_frozen` is None. "
                   "This is likely due to using the callback outside the trainer, "
                   "which automatically calls `setup()`.")
            logger.error(msg)
            raise ValueError(msg)

        train_mode = None
        for m in pl_module.model.modules():  # type: ignore[attr-defined]
            if isinstance(m, _BatchNorm):
                train_mode = self.train_bn or not any(self._are_frozen)
                m.train(train_mode)
        return train_mode