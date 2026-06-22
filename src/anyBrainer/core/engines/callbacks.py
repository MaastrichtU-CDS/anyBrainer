"""Define custom callbacks for Pytorch Lightning."""

from __future__ import annotations

__all__ = [
    "UpdateDatamoduleEpoch",
    "LogLR",
    "LogGradNorm",
    "FreezeParamGroups",
    "MetricAggregator",
    "SWAAvgOnly",
]

import logging
from typing import Any, Literal, cast
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only


from anyBrainer.registry import register, RegistryKind as RK
from anyBrainer.core.utils import (
    get_optimizer_lr,
    get_total_grad_norm,
    get_parameter_groups_from_prefixes,
)
from anyBrainer.core.engines.utils import (
    get_ckpt_callback,
)

logger = logging.getLogger(__name__)


@register(RK.CALLBACK)
class UpdateDatamoduleEpoch(pl.Callback):
    """Callback to update the datamodule's _current_epoch attribute."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Sync the current epoch between the trainer and the datamodule."""
        if trainer.datamodule is None or not hasattr(  # type: ignore[attr-defined]
            trainer.datamodule, "_current_epoch"  # type: ignore[attr-defined]
        ):
            logger.warning(
                "Datamodule does not have _current_epoch attribute. "
                "This is likely due to a custom datamodule that does not "
                "inherit from anyBrainer.data.BaseDataModule."
            )
            return

        trainer.datamodule._current_epoch = trainer.current_epoch + 1  # type: ignore
        logger.debug(f"Updated datamodule epoch to {trainer.current_epoch + 1}")


@register(RK.CALLBACK)
class LogLR(pl.Callback):
    """Log optimizer learning rates once per epoch."""

    @rank_zero_only
    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        metrics = get_optimizer_lr(trainer.optimizers)

        for trainer_logger in trainer.loggers:
            trainer_logger.log_metrics(
                metrics,
                step=trainer.global_step,
            )


@register(RK.CALLBACK)
class LogGradNorm(pl.Callback):
    """Callback to log the gradient norm for all parameters in the model."""

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        """Log gradient norm before clipping."""
        g = get_total_grad_norm(pl_module).detach()

        # Reduce across ranks ->  global L2: sqrt(sum_ranks(g^2))
        if dist.is_available() and dist.is_initialized():
            g2 = g.float().pow(2)
            dist.all_reduce(g2, op=dist.ReduceOp.SUM)
            g = g2.sqrt()

        pl_module.log(
            "train/grad_norm_preclip",
            g,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,  # already reduced
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log the gradient norm after clipping."""
        g = get_total_grad_norm(pl_module).detach()

        # Reduce across ranks ->  global L2: sqrt(sum_ranks(g^2))
        if dist.is_available() and dist.is_initialized():
            g2 = g.float().pow(2)
            dist.all_reduce(g2, op=dist.ReduceOp.SUM)
            g = g2.sqrt()

        pl_module.log(
            "train/grad_norm_postclip",
            g,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,  # already reduced
        )


@register(RK.CALLBACK)
class FreezeParamGroups(pl.Callback):
    """Freeze / unfreeze selected parameter groups at scheduled epochs or
    steps.

    Args:
        param_group_prefix: List of parameter group prefixes to freeze.
        freeze_epoch: Epoch at which to freeze (mutually exclusive with freeze_step).
        unfreeze_epoch: Epoch at which to unfreeze (mutually exclusive with unfreeze_step).
        freeze_step: Step at which to freeze (mutually exclusive with freeze_epoch).
        unfreeze_step: Step at which to unfreeze (mutually exclusive with unfreeze_epoch).
        train_bn: Whether to keep BatchNorm layers in train mode.
    """

    def __init__(
        self,
        *,
        param_group_prefix: str | list[str],
        freeze_epoch: int | list[int] | None = None,
        unfreeze_epoch: int | list[int] | None = None,
        freeze_step: int | list[int] | None = None,
        unfreeze_step: int | list[int] | None = None,
        train_bn: bool = False,
    ) -> None:
        super().__init__()

        # Determine mode: epoch-based or step-based
        use_epochs = freeze_epoch is not None or unfreeze_epoch is not None
        use_steps = freeze_step is not None or unfreeze_step is not None

        if use_epochs and use_steps:
            msg = (
                "[FreezeParamGroups] Cannot mix epoch-based and step-based scheduling."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.use_steps = use_steps

        # Normalise prefixes
        if isinstance(param_group_prefix, str):
            param_group_prefix = [param_group_prefix]
        self.prefixes = param_group_prefix

        # Normalise schedule values
        def _expand(x: int | list[int] | None, default: int = -1) -> list[int]:
            if x is None:
                return [default] * len(self.prefixes)
            return [x] * len(self.prefixes) if isinstance(x, int) else x

        if use_steps:
            self.freeze_at = _expand(freeze_step)
            self.unfreeze_at = _expand(unfreeze_step)
        else:
            self.freeze_at = _expand(freeze_epoch, default=0)
            self.unfreeze_at = _expand(unfreeze_epoch, default=0)

        if len(self.freeze_at) != len(self.prefixes):
            msg = (
                f"[FreezeParamGroups] freeze schedule must match number of parameter "
                f"group prefixes, got {len(self.freeze_at)} and {len(self.prefixes)}."
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(self.unfreeze_at) != len(self.prefixes):
            msg = (
                f"[FreezeParamGroups] unfreeze schedule must match number of parameter "
                f"group prefixes, got {len(self.unfreeze_at)} and {len(self.prefixes)}."
            )
            logger.error(msg)
            raise ValueError(msg)

        self.train_bn = train_bn
        self._are_frozen: list[bool] | None = None
        self._param_groups: list[list[nn.Parameter]] | None = None

    def setup(self, trainer, pl_module, stage: str | None = None):
        self._param_groups = [
            cast(
                list[nn.Parameter],
                get_parameter_groups_from_prefixes(
                    model=pl_module.model,  # type: ignore[attr-defined]
                    prefixes=pref,
                    trainable_only=False,
                    silent=True,
                ),
            )
            for pref in self.prefixes
        ]
        self._are_frozen = [False] * len(self._param_groups)

        mode = "steps" if self.use_steps else "epochs"
        logger.info(
            f"[FreezeParamGroups] managing {len(self._param_groups)} groups "
            f"(freeze at {self.freeze_at}, unfreeze at {self.unfreeze_at}, "
            f"mode={mode}, train_bn={self.train_bn})."
        )

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Freeze/unfreeze at epoch boundaries (epoch mode only)."""
        if self.use_steps:
            return
        self._check_and_toggle(trainer.current_epoch, pl_module, "epoch")

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx
    ) -> None:
        """Freeze/unfreeze at step boundaries (step mode only)."""
        if not self.use_steps:
            return
        self._check_and_toggle(trainer.global_step, pl_module, "step")

    def _check_and_toggle(
        self, current: int, pl_module: pl.LightningModule, mode: str
    ) -> None:
        """Check schedule and toggle requires_grad."""
        frozen, unfrozen = self._toggle_requires_grad(current)
        bn_mode = self._set_batchnorm_mode(pl_module)

        if frozen or unfrozen:
            msg = (
                f"[FreezeCallback] {mode} {current} | "
                f"froze: {frozen or '—'} | unfroze: {unfrozen or '—'}"
            )
            if bn_mode is not None:
                msg += f" | BatchNorm train={bn_mode}"
            logger.info(msg)

    def _toggle_requires_grad(self, current: int) -> tuple[list[str], list[str]]:
        """Freeze/unfreeze groups, return lists of prefix names that
        changed."""
        if self._param_groups is None or self._are_frozen is None:
            msg = "[FreezeParamGroups] setup() was not called."
            logger.error(msg)
            raise ValueError(msg)

        frozen, unfrozen = [], []
        for i, params in enumerate(self._param_groups):
            if current == self.freeze_at[i] and not self._are_frozen[i]:
                for p in params:
                    p.requires_grad = False
                self._are_frozen[i] = True
                frozen.append(self.prefixes[i])

            if current == self.unfreeze_at[i] and self._are_frozen[i]:
                for p in params:
                    p.requires_grad = True
                self._are_frozen[i] = False
                unfrozen.append(self.prefixes[i])

        return frozen, unfrozen

    def _set_batchnorm_mode(self, pl_module) -> bool | None:
        """Put BN layers in train/eval depending on frozen status."""
        if self._are_frozen is None:
            msg = "[FreezeParamGroups] setup() was not called."
            logger.error(msg)
            raise ValueError(msg)

        train_mode = None
        for m in pl_module.model.modules():
            if isinstance(m, _BatchNorm):
                train_mode = self.train_bn or not any(self._are_frozen)
                m.train(train_mode)
        return train_mode


@register(RK.CALLBACK)
class MetricAggregator(pl.Callback):
    """Accumulates metrics you `self.log()` during *any* evaluation loop
    (validation, test, validate-only) and prints mean ± std after each run.

    Args:
    - prefix: Only metrics whose names start with this prefix are aggregated
        (e.g. 'val_' or 'test_').  Set to '' to catch everything.
    """

    def __init__(self, prefix: str | list[str] = ""):
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
                    logger.warning(
                        f"[MetricAggregator] Skipping non-scalar tensor metric '{k}'."
                    )
                    continue
                v = v.item()
            self._current_run[k] = v

    def _summarise_run(self, tag: str) -> None:
        """Summarise and reset the collected metrics at the end of a run."""
        if not self._current_run:
            rank_zero_only(logger.warning)(
                f"[MetricAggregator] No '{tag}' metrics "
                f"collected for run {self._run_idx}."
            )
        else:
            lines = []
            for k, v in self._current_run.items():
                self._all_runs[k].append(v)
                lines.append(f"{k}: {v:.4f}")
            rank_zero_only(logger.info)(
                f"[MetricAggregator] {tag}-run {self._run_idx} "
                f"summary\n  " + "\n  ".join(lines)
            )
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
        if (
            ckpt_cb is not None
            and ckpt_cb.best_model_path
            and ckpt_cb.best_model_score is not None
        ):
            best_score = ckpt_cb.best_model_score.item()  # pyright: ignore
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


@register(RK.CALLBACK)
class SWAAvgOnly(pl.Callback):
    """Equal-weight parameter averaging without LR scheduling or BN updates.

    Under DDP, every rank maintains its own identical averaged model.
    """

    def __init__(
        self,
        start_epoch: int = 40,
        clamp_lr: float | None = None,
        update_on: Literal["epoch", "step"] = "epoch",
        average_device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        if update_on not in {"epoch", "step"}:
            raise ValueError(
                f"`update_on` must be 'epoch' or 'step', got {update_on!r}."
            )
        if start_epoch < 0:
            raise ValueError("`start_epoch` must be non-negative.")

        self.start_epoch = start_epoch
        self.clamp_lr = clamp_lr
        self.update_on = update_on
        self.average_device = average_device

        self._avg: AveragedModel | None = None
        self._active = False
        self._latest_update_step = 0
        self._latest_update_epoch = -1

        # Used when callback state is restored before AveragedModel exists.
        self._pending_avg_state: dict[str, Any] | None = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        if stage != "fit":
            return

        device = (
            torch.device(self.average_device)
            if self.average_device is not None
            else pl_module.device
        )

        # Do not unwrap the LightningModule. Lightning's official weight
        # averaging callback also passes pl_module directly to AveragedModel.
        self._avg = AveragedModel(
            pl_module,
            device=device,
            use_buffers=False,
        )

        if self._pending_avg_state is not None:
            self._avg.load_state_dict(self._pending_avg_state)
            self._pending_avg_state = None

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # >= is necessary when resuming after start_epoch.
        self._active = trainer.current_epoch >= self.start_epoch

        if not self._active or self.clamp_lr is None:
            return

        # Executed on every rank. Each rank owns a separate optimizer.
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(self.clamp_lr)

        if trainer.is_global_zero and trainer.current_epoch == self.start_epoch:
            logger.info(f"[SWAAvgOnly] Activated at epoch " f"{trainer.current_epoch}.")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.update_on != "step" or not self._active or self._avg is None:
            return

        # Prevent duplicate averaging under gradient accumulation because
        # several batches can share the same global_step.
        if trainer.global_step <= self._latest_update_step:
            return

        # Must execute on every rank.
        self._avg.update_parameters(pl_module)
        self._latest_update_step = trainer.global_step

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if (
            self.update_on != "epoch"
            or trainer.current_epoch < self.start_epoch
            or self._avg is None
            or trainer.current_epoch <= self._latest_update_epoch
        ):
            return

        # Must execute on every rank.
        self._avg.update_parameters(pl_module)
        self._latest_update_epoch = trainer.current_epoch

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._avg is None or int(self._avg.n_averaged.item()) == 0:
            if trainer.is_global_zero:
                logger.warning("[SWAAvgOnly] No models were averaged.")
            return

        # Every rank installs its own identical averaged parameters.
        pl_module.load_state_dict(self._avg.module.state_dict())

        if trainer.is_global_zero:
            logger.info(
                f"[SWAAvgOnly] Installed average of "
                f"{int(self._avg.n_averaged.item())} models."
            )

    def state_dict(self) -> dict[str, Any]:
        """Persist the running average for fault-tolerant resume."""
        return {
            "active": self._active,
            "latest_update_step": self._latest_update_step,
            "latest_update_epoch": self._latest_update_epoch,
            "average_model": (None if self._avg is None else self._avg.state_dict()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._active = bool(state_dict.get("active", False))
        self._latest_update_step = int(state_dict.get("latest_update_step", 0))
        self._latest_update_epoch = int(state_dict.get("latest_update_epoch", -1))

        average_state = state_dict.get("average_model")
        if average_state is None:
            return

        if self._avg is None:
            self._pending_avg_state = average_state
        else:
            self._avg.load_state_dict(average_state)
