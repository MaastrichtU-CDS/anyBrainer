"""Utility functions for models."""

__all__ = [
    "count_model_params",
    "summarize_model_params",
    "get_total_grad_norm",
    "get_optimizer_lr",
    "init_swin_with_residual_convs",
]

import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim

from anyBrainer.registry import register, RegistryKind as RK

logger = logging.getLogger(__name__)


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

def get_optimizer_lr(optimizers: list[optim.Optimizer]) -> dict[str, float]:
    """Get optimizer learning rates."""
    return {
        f"train/lr/opt{i}_group{j}": group["lr"]
        for i, opt in enumerate(optimizers)
        for j, group in enumerate(opt.param_groups)
    }

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
def init_swin_with_residual_convs(model: nn.Module,
                                  *,
                                  proj_std: float = 0.02,
                                  conv_mode: str = "fan_out",
                                  zero_init_residual: bool = False) -> None:
    """
    Weight initialisation for a Swin-style ViT that contains
    linear projection layers and (optionally 3 x 3) residual convolutions
    before each Swin block.

    Args:
        model : nn.Module
            Your complete Swin ViT model (or sub-module).
        proj_std : float, default 0.02
            Std for truncated-normal init of all nn.Linear weights (ViT default).
        conv_mode : {"fan_out", "fan_in"}, default "fan_out"
            Kaiming mode for convolutions. "fan_out" is common in modern conv nets.
        zero_init_residual : bool, default False
            If True, the *last* conv in every residual branch is zero-initialised,
            encouraging the residual path to start as identity.
    """
    # helper to decide whether a conv is "last" in a residual branch
    def _is_last_residual_conv(m: nn.Conv2d | nn.Conv3d) -> bool:
        # convention: last conv in your residual pre-conv stack ends with ".2"
        # e.g. 'residual_conv.2.weight'
        # Modify this if your naming differs or pass False to skip.
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