"""Factory functions for creating model, optimizer, LR scheduler, and loss function instances."""

__all__ = [
    "get_model_instance_from_kwargs",
    "get_optimizer_instances_from_kwargs",
    "get_lr_scheduler_instances_from_kwargs",
    "get_param_scheduler_instances_from_kwargs",
    "get_loss_fn_instances_from_kwargs",
]

import logging
from typing import Any

import torch.nn as nn
import torch.optim as optim

import anyBrainer.networks.nets as nets
import anyBrainer.schedulers.lr_schedulers as lr_schedulers
import anyBrainer.schedulers.param_schedulers as param_schedulers
import anyBrainer.losses as losses

logger = logging.getLogger(__name__)


def get_model_instance_from_kwargs(model_kwargs: dict[str, Any]) -> nn.Module:
    """
    Get model instance from kwargs.

    Model class is retrieved from anyBrainer.models.networks.
    
    Args:
        model_kwargs: dict[str, Any] - model kwargs. Must contain "name" key.

    Raises:
        ValueError: If model name is not found in model_kwargs.
        TypeError: If retrieved object is not a subclass of nn.Module.
        Exception: If error occurs during model initialization.
    """
    # Ensure required keys are provided
    if "name" not in model_kwargs:
        msg = "Model name not found in model_kwargs."
        logger.error(msg)
        raise ValueError(msg)
    
    model_kwargs = model_kwargs.copy()

    # Extract requested model class
    try:
        model_name = model_kwargs.pop("name")
        model_cls = getattr(nets, model_name)
    except AttributeError:
        msg = f"Model '{model_name}' not found in anyBrainer.models.networks."
        logger.error(msg)
        raise ValueError(msg)
    
    # Ensure model is a subclass of nn.Module
    if not issubclass(model_cls, nn.Module):
        msg = f"Retrieved object '{model_name}' is not a subclass of nn.Module."
        logger.error(msg)
        raise TypeError(msg)
    
    # Handle improper initialization args
    try:
        model = model_cls(**model_kwargs)
    except Exception as e:
        msg = f"Error initializing model '{model_name}': {e}"
        logger.exception(msg)
        raise
    
    return model

def get_optimizer_instances_from_kwargs(
    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
    model: nn.Module,
) -> optim.Optimizer | list[optim.Optimizer]:
    """
    Get optimizer instances from kwargs.

    Optimizer class is retrieved from torch.optim.

    If optimizer_kwargs is a list, return a list of optimizers through recursive calls.

    Args:
        optimizer_kwargs: dict[str, Any] - optimizer kwargs. Must contain "name" key.
        model: nn.Module - model to optimize.

   Raises:
        - ValueError: If optimizer name is not provided in optimizer_kwargs.
        - ValueError: If requested optimizer is not found in torch.optim.
        - Exception: If error occurs during optimizer initialization.
    """
    # Handle multiple optimizers with recursive calls
    if isinstance(optimizer_kwargs, list):
        optimizers_list = []
        for _optimizer_kwargs in optimizer_kwargs:
            optimizers_list.append(get_optimizer_instances_from_kwargs(_optimizer_kwargs, model))
        return optimizers_list

    # Ensure required keys are provided
    if "name" not in optimizer_kwargs:
        msg = "Optimizer name not found in optimizer_kwargs."
        logger.error(msg)
        raise ValueError(msg)

    optimizer_kwargs = optimizer_kwargs.copy()
    
    # Extract requested optimizer class
    try:
        cls_name = optimizer_kwargs.pop("name")
        optimizer_cls = getattr(optim, cls_name)
    except AttributeError:
        msg = f"Optimizer '{cls_name}' not found in torch.optim."
        logger.error(msg)
        raise ValueError(msg)
    
    # Handle improper initialization args
    try:
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    except Exception as e:
        msg = f"Error initializing optimizer '{cls_name}': {e}"
        logger.exception(msg)
        raise

    return optimizer

def get_lr_scheduler_instances_from_kwargs(
    lr_scheduler_kwargs: dict | list[dict],
    optimizer: optim.Optimizer | list[optim.Optimizer],
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Get LR scheduler instances from torch.optim.lr_scheduler or anyBrainer.models.schedulers.

    The returned LR scheduler is a dictionary with the instance and keys expected in Lightning's lr_scheduler_config.

    Special cases:
    - If multiple LR schedulers and optimizers are provided, return a list of lr_scheduler dictionaries.
    - If a single LR scheduler is provided with multiple optimizers, return a list of lr_scheduler
        dictionaries for each optimizer.

    Raises:
    - ValueError: If lr_scheduler_kwargs is a list and optimizer is not a list.
    - ValueError: If lr_scheduler_kwargs is a list and optimizer is a list but the lengths do not match.
    - ValueError: If lr_scheduler name is not provided in lr_scheduler_kwargs.
    - ValueError: If requested lr_scheduler is not found in torch.optim.lr_scheduler or
                    anyBrainer.models.schedulers.
    - ValueError: If lr_scheduler name, interval, and frequency are not provided in lr_scheduler_kwargs.
    - Exception: If error occurs during lr_scheduler initialization.
    """
    # Recursive calls to handle multiple LR schedulers
    if isinstance(lr_scheduler_kwargs, list):
        if not isinstance(optimizer, list):
            msg = "Optimizer must be a list if lr_scheduler_kwargs is a list."
            logger.error(msg)
            raise ValueError(msg)
        
        if len(lr_scheduler_kwargs) != len(optimizer):
            msg = "Length of lr_scheduler_kwargs and optimizer must match."
            logger.error(msg)
            raise ValueError(msg)
        
        schedulers_list = []
        for _scheduler, _optimizer in zip(lr_scheduler_kwargs, optimizer):
            schedulers_list.append(get_lr_scheduler_instances_from_kwargs(_scheduler, _optimizer))
        return schedulers_list
    
    if not isinstance(lr_scheduler_kwargs, list) and isinstance(optimizer, list):
        schedulers_list = []
        for _optimizer in optimizer:
            schedulers_list.append(get_lr_scheduler_instances_from_kwargs(lr_scheduler_kwargs, _optimizer))
        return schedulers_list
    
    # Ensure required keys are provided
    if "name" not in lr_scheduler_kwargs:
        msg = "Scheduler name not found in lr_scheduler_kwargs."
        logger.error(msg)
        raise ValueError(msg)
    
    required_keys = {"name", "interval", "frequency"}
    if required_keys.issubset(lr_scheduler_kwargs.keys()):
        lr_scheduler_kwargs = lr_scheduler_kwargs.copy()
        lr_scheduler_dict = {
            "name": lr_scheduler_kwargs.pop("name"),
            "interval": lr_scheduler_kwargs.pop("interval"),
            "frequency": lr_scheduler_kwargs.pop("frequency"),
            "monitor": lr_scheduler_kwargs.pop("monitor", None),
            "strict": lr_scheduler_kwargs.pop("strict", False),
        }
    else:
        msg = "LR scheduler 'name', 'interval', and 'frequency' must be provided."
        logger.error(msg)
        raise ValueError(msg)
    
    # Extract requested LR scheduler class
    try:
        lr_scheduler_cls = getattr(optim.lr_scheduler, lr_scheduler_dict["name"])
    except AttributeError:
        try:
            lr_scheduler_cls = getattr(lr_schedulers, lr_scheduler_dict["name"])
        except AttributeError:  
            msg = (f"LR scheduler '{lr_scheduler_dict['name']}' not found in "
                    f"torch.optim.lr_scheduler or anyBrainer.models.schedulers.")
            logger.error(msg)
            raise ValueError(msg)
    
    # Handle improper initialization args
    try:
        lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)
        lr_scheduler_dict["scheduler"] = lr_scheduler
    except Exception as e:
        msg = f"Error initializing LR scheduler '{lr_scheduler_dict['name']}': {e}"
        logger.exception(msg)
        raise 

    return lr_scheduler_dict

def get_param_scheduler_instances_from_kwargs(
    other_schedulers: list[dict[str, Any]],
) -> tuple[list[param_schedulers.ParameterScheduler], list[param_schedulers.ParameterScheduler]]:
    """
    Get any other custom schedulers from anyBrainer.models.schedulers.
    
    Groups the schedulers into step and epoch schedulers, the values of which are
    extracted using built-in Lightning hooks.
    Always assuming list of dicts.

    Raises:
    - ValueError: If scheduler name is not provided in scheduler_kwargs.
    - ValueError: If scheduler interval is not found in scheduler_kwargs.
    - ValueError: If requested scheduler is not found in anyBrainer.models.schedulers.
    - Exception: If error occurs during scheduler initialization.
    """
    other_schedulers_step = []
    other_schedulers_epoch = []
    for scheduler_kwargs in other_schedulers:
        # Ensure required keys are provided
        if "name" not in scheduler_kwargs:
            msg = "Scheduler name not found in scheduler_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        if "interval" not in scheduler_kwargs:
            msg = "Scheduler interval not found in scheduler_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        scheduler_kwargs = scheduler_kwargs.copy()
        
        # Extract requested scheduler class
        try:
            scheduler_name = scheduler_kwargs.pop("name")
            scheduler_interval = scheduler_kwargs.pop("interval")
            scheduler_cls = getattr(param_schedulers, scheduler_name)
        except AttributeError:
            msg = f"Scheduler '{scheduler_name}' not found in anyBrainer.models.schedulers."
            logger.error(msg)
            raise ValueError(msg)   
        
        # Handle improper initialization args
        try:
            scheduler = scheduler_cls(**scheduler_kwargs)
        except Exception as e:
            msg = f"Error initializing scheduler '{scheduler_name}': {e}"
            logger.exception(msg)
            raise
        
        # Add scheduler to list
        if scheduler_interval == "step":
            other_schedulers_step.append(scheduler)
        elif scheduler_interval == "epoch":
            other_schedulers_epoch.append(scheduler)
        else:
            msg = f"Scheduler interval '{scheduler_interval}' not supported."
            logger.error(msg)
            raise ValueError(msg)
    
    return other_schedulers_step, other_schedulers_epoch

def get_loss_fn_instances_from_kwargs(
    loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
) -> nn.Module | list[nn.Module]:
    """
    Get loss function instances from torch.nn or anyBrainer.models.losses.

    Special cases:
    - If multiple loss functions are provided, return a list of loss function instances.

    Raises:
    - ValueError: If loss fn name is not found in loss_fn_kwargs.
    - ValueError: If loss fn name is not found in torch.nn or anyBrainer.models.losses.
    - Exception: If error occurs during loss fn initialization.
    """
    # Handle multiple optimizers with recursive calls
    if isinstance(loss_fn_kwargs, list):
        loss_fns_list = []
        for _loss_fn_kwargs in loss_fn_kwargs:
            loss_fns_list.append(get_loss_fn_instances_from_kwargs(_loss_fn_kwargs))
        return loss_fns_list

    # Ensure required keys are provided
    if "name" not in loss_fn_kwargs:
        msg = "Loss fn name not found in loss_fn_kwargs."
        logger.error(msg)
        raise ValueError(msg)

    loss_fn_kwargs = loss_fn_kwargs.copy()
    
    # Extract requested loss fn class
    try:
        loss_fn_name = loss_fn_kwargs.pop("name")
        loss_fn_cls = getattr(nn, loss_fn_name)
    except AttributeError:
        try:
            loss_fn_cls = getattr(losses, loss_fn_name)
        except AttributeError:  
            msg = (f"Loss fn '{loss_fn_name}' not found in torch.nn or "
                    f"anyBrainer.models.losses.")
            logger.error(msg)
            raise ValueError(msg)
    
    # Handle improper initialization args
    try:
        loss_fn = loss_fn_cls(**loss_fn_kwargs)
    except Exception as e:
        msg = f"Error initializing loss fn '{loss_fn_kwargs['name']}': {e}"
        logger.exception(msg)
        raise

    return loss_fn