"""Factory for creating unit (nets, optimizers, schedulers, etc.) instances."""


from __future__ import annotations

__all__ = [
    "UnitFactory",
]

import logging
from typing import Any, Callable, TYPE_CHECKING

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import lightning.pytorch.callbacks as pl_callbacks
import monai.inferers as monai_inferers
import monai.networks.nets as monai_nets

from anyBrainer.registry import get, RegistryKind as RK

if TYPE_CHECKING:
    from monai.inferers.inferer import Inferer
    from anyBrainer.interfaces import ParameterScheduler

logger = logging.getLogger(__name__)


class UnitFactory:
    @classmethod
    def get_model_instance_from_kwargs(
        cls,
        model_kwargs: dict[str, Any],
    ) -> nn.Module:
        """
        Get model instance from kwargs.

        Model class is retrieved from anyBrainer.registry-NETWORK or monai.networks.nets.

        Args:
            model_kwargs: model kwargs, containing "name" key.
        
        Raises:
            ValueError: If model name is not provided in model_kwargs.
            ValueError: If requested model is not found in anyBrainer.registry-NETWORK.
            Exception: If error occurs during model initialization.
        """
        # Ensure required keys are provided
        if "name" not in model_kwargs:
            msg = "Model name not found in model_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        model_kwargs = model_kwargs.copy()
        model_name = model_kwargs.pop("name")

        # Extract requested model class
        try:
            model_cls = get(RK.NETWORK, model_name)
        except ValueError:
            try:
                model_cls = getattr(monai_nets, model_name)
            except AttributeError:
                msg = (f"Model '{model_name}' not found in anyBrainer.registry-NETWORK "
                       "or monai.networks.nets.")
                logger.error(msg)
                raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            model = model_cls(**model_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing model '{model_name}': {e}"
            logger.exception(msg)
            raise
        
        return model

    @classmethod
    def get_optimizer_instances_from_kwargs(
        cls,
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
    ) -> optim.Optimizer | list[optim.Optimizer]:
        """
        Get optimizer instances from kwargs.

        Optimizer class is retrieved from torch.optim.
        Expects users to provide `params`: list[nn.Parameter] | Iterable[nn.Parameter]
        inside `optimizer_kwargs`.

        If `optimizer_kwargs` is a list, return a list of optimizers through recursive calls.

        Args:
            optimizer_kwargs: optimizer kwargs, containing `name` key.

        Raises:
            - ValueError: If optimizer `name` is not provided in optimizer_kwargs.
            - ValueError: If `params` is not provided in optimizer_kwargs.
            - ValueError: If requested optimizer is not found in `torch.optim`.
            - Exception: If error occurs during optimizer initialization.
        """
        # Handle multiple optimizers with recursive calls
        if isinstance(optimizer_kwargs, list):
            optimizers_list = []
            for _optimizer_kwargs in optimizer_kwargs:
                optimizers_list.append(cls.get_optimizer_instances_from_kwargs(_optimizer_kwargs))
            return optimizers_list

        # Ensure required keys are provided
        if "name" not in optimizer_kwargs:
            msg = "Optimizer name not found in optimizer_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        if "params" not in optimizer_kwargs:
            msg = "`params` not found in optimizer_kwargs."
            logger.error(msg)
            raise ValueError(msg)

        optimizer_kwargs = optimizer_kwargs.copy()
        cls_name = optimizer_kwargs.pop("name")
        params = optimizer_kwargs.pop("params")
        
        # Extract requested optimizer class
        try:
            optimizer_cls = getattr(optim, cls_name)
        except AttributeError:
            msg = f"Optimizer '{cls_name}' not found in torch.optim."
            logger.error(msg)
            raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            optimizer = optimizer_cls(params, **optimizer_kwargs)
        except Exception as e:
            msg = f"Error initializing optimizer '{cls_name}': {e}"
            logger.exception(msg)
            raise

        return optimizer

    @classmethod
    def get_lr_scheduler_instances_from_kwargs(
        cls,
        lr_scheduler_kwargs: dict | list[dict],
        optimizer: optim.Optimizer | list[optim.Optimizer],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Get LR scheduler instances from torch.optim.lr_scheduler or anyBrainer.registry-LR_SCHEDULER.

        The returned LR scheduler is a dictionary with the instance and keys expected in Lightning's lr_scheduler_config.

        Special cases:
        - If multiple LR schedulers and optimizers are provided, return a list of lr_scheduler dictionaries.
        - If a single LR scheduler is provided with multiple optimizers, return a list of lr_scheduler
            dictionaries for each optimizer.

        Args:
            lr_scheduler_kwargs: lr scheduler kwargs, containing "name", "interval", and "frequency" keys.
            optimizer: optimizer or list of optimizers.

        Raises:
        - ValueError: If lr_scheduler_kwargs is a list and optimizer is not a list.
        - ValueError: If lr_scheduler_kwargs is a list and optimizer is a list but the lengths do not match.
        - ValueError: If lr_scheduler name is not provided in lr_scheduler_kwargs.
        - ValueError: If requested lr_scheduler is not found in torch.optim.lr_scheduler or
                        anyBrainer.registry-LR_SCHEDULER.
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
                schedulers_list.append(cls.get_lr_scheduler_instances_from_kwargs(_scheduler, _optimizer))
            return schedulers_list

        if not isinstance(lr_scheduler_kwargs, list) and isinstance(optimizer, list):
            schedulers_list = []
            for _optimizer in optimizer:
                schedulers_list.append(cls.get_lr_scheduler_instances_from_kwargs(lr_scheduler_kwargs, _optimizer))
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
                lr_scheduler_cls = get(RK.LR_SCHEDULER, lr_scheduler_dict["name"])
            except ValueError:  
                msg = (f"LR scheduler '{lr_scheduler_dict['name']}' not found in "
                        f"torch.optim.lr_scheduler or anyBrainer.registry-LR_SCHEDULER.")
                logger.error(msg)
                raise ValueError(msg)

        # Handle improper initialization args
        try:
            lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs) # type: ignore
            lr_scheduler_dict["scheduler"] = lr_scheduler
        except Exception as e:
            msg = f"Error initializing LR scheduler '{lr_scheduler_dict['name']}': {e}"
            logger.exception(msg)
            raise 

        return lr_scheduler_dict

    @classmethod
    def get_param_scheduler_instances_from_kwargs(
        cls,
        other_schedulers: list[dict[str, Any]],
    ) -> tuple[list[ParameterScheduler], list[ParameterScheduler]]:
        """
        Get any other custom schedulers from anyBrainer.registry-PARAM_SCHEDULER.
        
        Groups the schedulers into step and epoch schedulers, the values of which are
        extracted using built-in Lightning hooks.

        Args:
            other_schedulers: list of scheduler kwargs, each containing "name" and "interval" keys.

        Raises:
        - ValueError: If scheduler name is not provided in scheduler_kwargs.
        - ValueError: If scheduler interval is not found in scheduler_kwargs.
        - ValueError: If requested scheduler is not found in anyBrainer.registry-PARAM_SCHEDULER.
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
            scheduler_name = scheduler_kwargs.pop("name")
            scheduler_interval = scheduler_kwargs.pop("interval")
            
            try:
                scheduler_cls = get(RK.PARAM_SCHEDULER, scheduler_name)
            except ValueError:
                msg = f"Scheduler '{scheduler_name}' not found in anyBrainer.registry-PARAM_SCHEDULER."
                logger.error(msg)
                raise ValueError(msg)   
            
            # Handle improper initialization args
            try:
                scheduler = scheduler_cls(**scheduler_kwargs) # type: ignore
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

    @classmethod
    def get_loss_fn_instances_from_kwargs(
        cls,
        loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]],
    ) -> nn.Module | list[nn.Module]:
        """
        Get loss function instances from torch.nn or anyBrainer.registry-LOSS.

        Special cases:
        - If multiple loss functions are provided, return a list of loss function instances.

        Args:
            loss_fn_kwargs: loss fn kwargs, containing "name" key.

        Raises:
        - ValueError: If loss fn name is not found in loss_fn_kwargs.
        - ValueError: If loss fn name is not found in torch.nn or anyBrainer.registry-LOSS.
        - Exception: If error occurs during loss fn initialization.
        """
        # Handle multiple optimizers with recursive calls
        if isinstance(loss_fn_kwargs, list):
            loss_fns_list = []
            for _loss_fn_kwargs in loss_fn_kwargs:
                loss_fns_list.append(cls.get_loss_fn_instances_from_kwargs(_loss_fn_kwargs))
            return loss_fns_list

        # Ensure required keys are provided
        if "name" not in loss_fn_kwargs:
            msg = "Loss fn name not found in loss_fn_kwargs."
            logger.error(msg)
            raise ValueError(msg)

        loss_fn_kwargs = loss_fn_kwargs.copy()
        loss_fn_name = loss_fn_kwargs.pop("name")
        
        # Extract requested loss fn class
        try:
            loss_fn_cls = getattr(nn, loss_fn_name)
        except AttributeError:
            try:
                loss_fn_cls = get(RK.LOSS, loss_fn_name)
            except ValueError:  
                msg = (f"Loss fn '{loss_fn_name}' not found in torch.nn or "
                        f"anyBrainer.registry-LOSS.")
                logger.error(msg)
                raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            loss_fn = loss_fn_cls(**loss_fn_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing loss fn '{loss_fn_name}': {e}"
            logger.exception(msg)
            raise

        return loss_fn

    @classmethod
    def get_pl_callback_instances_from_kwargs(
        cls,
        callback_kwargs: list[dict[str, Any]],
    ) -> list[pl.Callback]:
        """
        Get callback instances from anyBrainer.registry-CALLBACK or pl.callbacks. 

        Args:
            callback_kwargs: list of callback kwargs, each containing "name" key.

        Raises:
        - ValueError: If callback name is not found in callback_kwargs.
        - ValueError: If requested callback is not found in anyBrainer.registry-CALLBACK or pl.callbacks.
        - Exception: If error occurs during callback initialization.
        """
        callbacks_list = []
        for _callback_kwargs in callback_kwargs:
            # Ensure required keys are provided
            if "name" not in _callback_kwargs:
                msg = "Callback name not found in callback_kwargs."
                logger.error(msg)
                raise ValueError(msg)
            
            _callback_kwargs = _callback_kwargs.copy()
            callback_name = _callback_kwargs.pop("name")
            
            # Extract requested callback class
            try:
                callback_cls = getattr(pl_callbacks, callback_name)
            except AttributeError:
                try:
                    callback_cls = get(RK.CALLBACK, callback_name)
                except ValueError:
                    msg = (f"Callback '{callback_name}' not found in anyBrainer.registry-CALLBACK "
                        "or pl.callbacks.")
                    logger.error(msg)
                    raise ValueError(msg)
            
            try: # Handle improper initialization args
                callbacks_list.append(callback_cls(**_callback_kwargs)) # type: ignore
            except Exception as e:
                msg = f"Error initializing callback '{callback_name}': {e}"
                logger.exception(msg)
                raise
                
        return callbacks_list
    
    @classmethod
    def get_inferer_instance_from_kwargs(
        cls,
        inferer_kwargs: dict[str, Any],
    ) -> Inferer:
        """
        Get inferer instance from anyBrainer.registry-INFERER.

        Args:
            inferer_kwargs: inferer kwargs, containing "name" key.

        Raises:
        - ValueError: If inferer name is not provided in inferer_kwargs.
        - ValueError: If requested inferer is not found in anyBrainer.registry-INFERER or monai.inferers.
        - Exception: If error occurs during inferer initialization.
        """
        # Ensure required keys are provided
        if "name" not in inferer_kwargs:
            msg = "Inferer name not provided in inferer_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        inferer_kwargs = inferer_kwargs.copy()
        inferer_name = inferer_kwargs.pop("name")

        # Extract requested inferer class
        try:
            inferer_cls = getattr(monai_inferers, inferer_name)
        except AttributeError:
            try:
                inferer_cls = get(RK.INFERER, inferer_name)
            except ValueError:
                msg = (f"Inferer '{inferer_name}' not found in anyBrainer.registry-INFERER "
                       "or monai.inferers.")
                logger.error(msg)
                raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            inferer = inferer_cls(**inferer_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing inferer '{inferer_name}': {e}"
            logger.exception(msg)
            raise

        return inferer
    
    @classmethod
    def get_transformslist_from_kwargs(
        cls,
        transform_kwargs: dict[str, Any],
    ) -> list[Callable]: 
        """
        Get list of transforms from anyBrainer.registry-TRANSFORM, using a transform builder fn. 

        Args:
            transform_kwargs: transform kwargs, containing "name" key.

        Raises:
        - ValueError: If transform name is not present in transform_kwargs.
        - ValueError: If requested transform is not found in anyBrainer.registry-TRANSFORM.
        - Exception: If error occurs during transform initialization.
        """
        # Ensure required keys are provided
        if "name" not in transform_kwargs:
            msg = "Transform name not provided in transform_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        transform_kwargs = transform_kwargs.copy()
        transforms_name = transform_kwargs.pop("name")

        try:
            get_transforms_fn = get(RK.TRANSFORM, transforms_name)
        except ValueError:
            msg = f"Transform builder name '{transforms_name}' not found in anyBrainer.transforms."
            logger.error(msg)
            raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            transformslist = get_transforms_fn(**transform_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error retrieving transormslist '{transforms_name}': {e}"
            logger.exception(msg)
            raise

        return transformslist

    @classmethod
    def get_activation_from_kwargs(
        cls,
        activation_fn_kwargs: dict[str, Any],
    ) -> nn.Module:
        """
        Get activation function from torch.nn.

        Raises:
        - ValueError: If activation function name is not provided in activation_fn_kwargs.
        - ValueError: If requested activation function is not found in torch.nn.
        - Exception: If error occurs during activation function initialization.
        """
        if "name" not in activation_fn_kwargs:
            msg = "Activation function name not provided in activation_fn_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        activation_fn_kwargs = activation_fn_kwargs.copy()
        activation_fn_name = activation_fn_kwargs.pop("name")
        
        try:
            activation_fn_cls = getattr(nn, activation_fn_name)
        except AttributeError:
            msg = f"Activation function '{activation_fn_name}' not found in torch.nn."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            activation_fn = activation_fn_cls(**activation_fn_kwargs) # type: ignore
        except Exception as e:
            msg = f"Error initializing activation function '{activation_fn_name}': {e}"
            logger.exception(msg)
            raise
        
        return activation_fn