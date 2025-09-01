"""Utility functions for models."""

__all__ = [
    "count_model_params",
    "summarize_model_params",
    "get_total_grad_norm",
    "init_swin_v2",
    "get_optimizer_lr",
    "get_parameter_groups_from_prefixes",
    "split_decay_groups_from_params",
]

import logging
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from anyBrainer.registry import register, RegistryKind as RK

logger = logging.getLogger(__name__)

_NO_DECAY_NORM_TYPES = (
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)

# Common Swin/ViT params that should not decay
_NO_DECAY_NAME_KEYWORDS = {
    "pos_embed",                       # ViT / Swin absolute pos embed
    "absolute_pos_embed",              # some impls
    "relative_position_bias_table",    # Swin v1/v2
    "rel_pos_bias",                    # alt naming
    "logit_scale",                     # sometimes present in CLIP-like heads
    "gamma", "beta",                   # norm-style params in some impls
}

def count_model_params(model: nn.Module, trainable: bool = False):
    """Count model parameters."""
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def summarize_model_params(model: nn.Module) -> str:
    """Show summary of all model parameters and buffers."""
    out_msg = "\n#Model parameters#\n"
    for name, param in model.named_parameters():
        out_msg += f"{name:60s} | shape={tuple(param.shape)} | requires_grad={param.requires_grad}\n"
        
    out_msg += f"\n#Model buffers#\n"
    for name, b in model.named_buffers():
        out_msg += f"{name:55s} {tuple(b.shape)}\n"
        
    out_msg += f"\n#Model summary#\n"
    out_msg += f"Total parameters: {count_model_params(model)}\n"
    out_msg += f"Trainable parameters: {count_model_params(model, trainable=True)}\n"
    out_msg += f"Non-trainable parameters: {count_model_params(model, trainable=False)}\n"
    
    return out_msg

def get_total_grad_norm(model: nn.Module) -> torch.Tensor:
    """Get total gradient norm."""
    total_norm = torch.norm(
        torch.stack([
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]), p=2,
    )
    return total_norm

def _trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.,
                   a: float = -2., b: float = 2.) -> torch.Tensor:
    """
    Fills `tensor` with values drawn from a truncated N(mean, std) distribution.
    Modifies `tensor` in-place.
    """
    def norm_cdf(x: float) -> float:
        """Cumulative distribution function for the standard normal."""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    with torch.no_grad():
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

    return tensor

@register(RK.UTIL)
def init_swin_v2(model: nn.Module,
                                  *,
                                  proj_std: float = 0.02,
                                  conv_mode: str = "fan_out",
                                  zero_init_residual: bool = False) -> None:
    """
    Weight initialisation for a Swinv2-style ViT that contains
    linear projection layers and (optionally 3 x 3) residual convolutions
    before each Swinv2 block.

    Args:
        model : nn.Module
            Complete Swinv2 ViT model (or sub-module).
        proj_std : float, default 0.02
            Std for truncated-normal init of all nn.Linear weights (Swinv2 default).
        conv_mode : {"fan_out", "fan_in"}, default "fan_out"
            Kaiming mode for convolutions. "fan_out" is common in modern conv nets.
        zero_init_residual : bool, default False
            If True, the *last* conv in every residual branch is zero-initialised,
            encouraging the residual path to start as identity.
    """
    # helper to decide whether a conv is "last" in a residual branch
    def _is_last_residual_conv(m: nn.Conv2d | nn.Conv3d) -> bool:
        # convention: last conv in residual conv stack ends with ".2"
        # e.g. 'residual_conv.2.weight'
        return m.weight.shape[0] == m.weight.shape[1] and m.kernel_size == (3, 3)

    for _, m in model.named_modules():
        # Convolution layers
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if zero_init_residual and _is_last_residual_conv(m):
                nn.init.zeros_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode=conv_mode, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Linear (projection) layers
        elif isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=proj_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Normalisation layers
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                            nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def get_optimizer_lr(optimizers: list[optim.Optimizer]) -> dict[str, float]:
    """Get optimizer learning rates."""
    return {
        f"train/lr/opt{i}_group{j}": group["lr"]
        for i, opt in enumerate(optimizers)
        for j, group in enumerate(opt.param_groups)
    }

def get_parameter_groups_from_prefixes(
    model: nn.Module,
    prefixes: str | list[str] | None = None,
    *,
    trainable_only: bool = True,
    silent: bool = False,
    return_named: bool = False,
) -> list[tuple[str, nn.Parameter]] | list[nn.Parameter]:
    if prefixes is None:
        named = [
            (n, p) for n, p in model.named_parameters()
            if (p.requires_grad or not trainable_only)
        ]
    else:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        named = [
            (n, p) for n, p in model.named_parameters()
            if any(n.startswith(pref) for pref in prefixes)
            and (p.requires_grad or not trainable_only)
        ]
    if not silent and not named:
        msg = f"No parameters found for prefixes: {prefixes}"
        logger.error(msg)
        raise ValueError(msg)

    return named if return_named else [p for _, p in named]

def is_no_decay_param(
    name: str,
    module: nn.Module | None,
    *,
    extra_prefixes: Sequence[str] | None,
    auto_no_weight_decay: bool,
) -> bool:
    """
    Determine if a parameter should not receive weight decay.

    Args:
        name: The name of the parameter.
        module: The module instance of the parameter.
        extra_prefixes: Extra prefixes to check.
        auto_no_weight_decay: If True, weight decay is set to 0 for all parameters
        that are not explicitly listed in `no_weight_decay_prefixes`.
    """
    # Normalize names
    name = name.lower()
    extra_prefixes = tuple(p.lower() for p in extra_prefixes) if extra_prefixes else None
    
    if extra_prefixes and name.startswith(extra_prefixes):
        return True

    if not auto_no_weight_decay:
        return False

    # module-type rule: any Norm* module's params -> no decay
    if module is not None and isinstance(module, _NO_DECAY_NORM_TYPES):
        return True

    # bias parameters
    if name.endswith(".bias"):
        return True

    # name-based rules
    if any(kw in name for kw in _NO_DECAY_NAME_KEYWORDS):
        return True

    return False

def split_decay_groups_from_params(
    model: nn.Module,
    params: list[tuple[str, nn.Parameter]],
    *,
    auto_no_weight_decay: bool,
    no_weight_decay_prefixes: list[str] | None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (decay_params, no_decay_params) from a flat named param list."""
    if not auto_no_weight_decay and not no_weight_decay_prefixes:
        return [p for _, p in params], []

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in params:
        mod_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(mod_name) if mod_name else model
        if is_no_decay_param(name, module, extra_prefixes=no_weight_decay_prefixes, 
                             auto_no_weight_decay=auto_no_weight_decay):
            no_decay.append(p)   # append the Parameter
        else:
            decay.append(p)
    return decay, no_decay