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
from collections import defaultdict

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
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
    get_ckpt_callback,
)

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


@register(RK.CALLBACK)
class UpdateDatamoduleEpoch(pl.Callback):
    """
    Callback to update the datamodule's _current_epoch attribute.
    """        
    @rank_zero_only
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
    @rank_zero_only
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
    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        """Log the gradient norm."""
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        if not params:
            return
        
        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in params]), p=2,
        )
        pl_module.log(
            "train/grad_norm", 
            total_norm, 
            on_step=True, 
            on_epoch=False, 
            prog_bar=False, 
            sync_dist=False,
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


@register(RK.CALLBACK)
class MetricAggregator(pl.Callback):
    """
    Accumulates metrics you `self.log()` during *any* evaluation loop
    (validation, test, validate-only) and prints mean ± std after each run.

    Args:
    - prefix: Only metrics whose names start with this prefix are aggregated
        (e.g. 'val_' or 'test_').  Set to '' to catch everything.
    """

    def __init__(self, prefix: str | list[str] = ''):
        super().__init__()
        self.prefix = [prefix] if isinstance(prefix, str) else prefix
        self._current_run: dict[str, float] = defaultdict(float)
        self._all_runs: dict[str, list[float]] = defaultdict(list)
        self._run_idx = 0

    def _collect(self, trainer: pl.Trainer) -> None:
        """Collect filtered metrics from trainer.callback_metrics."""
        if trainer.sanity_checking:
            return
        for k, v in trainer.callback_metrics.items():
            if not any(k.startswith(p) for p in self.prefix):
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim != 0:
                    logger.warning(f"[MetricAggregator] Skipping non-scalar tensor metric '{k}'.")
                    continue
                v = v.item()
            self._current_run[k] = v

    def _summarise_run(self, tag: str) -> None:
        """Summarise and reset the collected metrics at the end of a run."""
        if not self._current_run:
            rank_zero_only(logger.warning)(f"[MetricAggregator] No '{tag}' metrics "
                                           f"collected for run {self._run_idx}.")
        else:
            lines = []
            for k, v in self._current_run.items():
                self._all_runs[k].append(v)
                lines.append(f"{k}: {v:.4f}")
            rank_zero_only(logger.info)(f"[MetricAggregator] {tag}-run {self._run_idx} "
                                        f"summary\n  " + "\n  ".join(lines))
        self._current_run.clear()
        self._run_idx += 1
    
    def on_validation_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ) -> None:
        self._collect(trainer)

    def on_test_epoch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ) -> None:
        self._collect(trainer)

    def on_fit_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ) -> None:
        # Add best checkpoint metric--if any--to current run.
        ckpt_cb = get_ckpt_callback(trainer)
        if ckpt_cb is not None and ckpt_cb.best_model_path:
            best_score = ckpt_cb.best_model_score.item() # pyright: ignore
            best_metric = ckpt_cb.monitor
            self._current_run[f"best_{best_metric}"] = best_score
        self._summarise_run("fit")

    def on_test_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
    ) -> None:
        self._summarise_run("test")

    @rank_zero_only
    def final_summary(self) -> None:
        """Call manually at the end of all runs to get global mean ± std."""
        if not self._all_runs:
            logger.warning("[MetricAggregator] No metrics to summarise across runs.")
            return

        logger.info("[MetricAggregator] === FINAL SUMMARY across all runs ===")
        for k, means in self._all_runs.items():
            t = torch.tensor(means)
            logger.info(f"{k}: {t.mean().item():.4f} ± {t.std().item():.4f}")