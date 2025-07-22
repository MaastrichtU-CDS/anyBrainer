"""Utility functions for models."""

__all__ = [
    "count_model_params",
    "summarize_model_params",
]

import torch
import torch.nn as nn


def count_model_params(model: nn.Module, trainable: bool = False):
    """Count model parameters."""
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def summarize_model_params(model: nn.Module) -> str:
    """Show summary of all model parameters and buffers."""
    out_msg = "#Model parameters#\n"
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