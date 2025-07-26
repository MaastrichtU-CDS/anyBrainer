"""
Factory module to instantiate commonly used objects. 

Currently, the factory module is used to create:
- unit (nets, optimizers, schedulers, etc.) instances.
- module (PL module, PL datamodule, managers) instances.

Optionally returns only the class without instantiating.
"""

__all__ = [
    "UnitFactory",
    "ModuleFactory",
]

import logging
from typing import Any, Callable

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks

import anyBrainer.networks.nets as nets
import anyBrainer.schedulers.lr_schedulers as lr_schedulers
import anyBrainer.schedulers.param_schedulers as param_schedulers
import anyBrainer.losses as losses
import anyBrainer.engines.models as models
import anyBrainer.data.datamodule as datamodules
import anyBrainer.engines.callbacks as callbacks
import anyBrainer.transforms.flat as transforms
import anyBrainer.log.logging_manager as logging_managers

logger = logging.getLogger(__name__)


class UnitFactory:
    @classmethod
    def get_model_instance_from_kwargs(
        cls,
        model_kwargs: dict[str, Any],
        cls_only: bool = False,
    ) -> nn.Module | type[nn.Module]:
        """
        Get model instance from kwargs.

        Model class is retrieved from anyBrainer.models.networks.
        
        Args:
            model_kwargs: dict[str, Any] - model kwargs. Must contain "name" key.
            cls_only: bool - whether to return only the class or the instance.

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

            if cls_only:
                return model_cls

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

    @classmethod
    def get_optimizer_instances_from_kwargs(
        cls,
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
        model: nn.Module,
        cls_only: bool = False,
    ) -> type[optim.Optimizer] | optim.Optimizer | list[type[optim.Optimizer]] | list[optim.Optimizer]:
        """
        Get optimizer instances from kwargs.

        Optimizer class is retrieved from torch.optim.

        If optimizer_kwargs is a list, return a list of optimizers through recursive calls.

        Args:
            optimizer_kwargs: dict[str, Any] - optimizer kwargs. Must contain "name" key.
            model: nn.Module - model to optimize.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
            - ValueError: If optimizer name is not provided in optimizer_kwargs.
            - ValueError: If requested optimizer is not found in torch.optim.
            - Exception: If error occurs during optimizer initialization.
        """
        # Handle multiple optimizers with recursive calls
        if isinstance(optimizer_kwargs, list):
            optimizers_list = []
            for _optimizer_kwargs in optimizer_kwargs:
                optimizers_list.append(cls.get_optimizer_instances_from_kwargs(_optimizer_kwargs, model))
            return optimizers_list

        # Ensure required keys are provided
        if "name" not in optimizer_kwargs:
            msg = "Optimizer name not found in optimizer_kwargs."
            logger.error(msg)
            raise ValueError(msg)

        optimizer_kwargs = optimizer_kwargs.copy()
        cls_name = optimizer_kwargs.pop("name")
        
        # Extract requested optimizer class
        try:
            optimizer_cls = getattr(optim, cls_name)

            if cls_only:
                return optimizer_cls

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

    @classmethod
    def get_lr_scheduler_instances_from_kwargs(
        cls,
        lr_scheduler_kwargs: dict | list[dict],
        optimizer: optim.Optimizer | list[optim.Optimizer],
        cls_only: bool = False,
    ) -> type[optim.lr_scheduler.LRScheduler] | dict[str, Any] | list[type[optim.lr_scheduler.LRScheduler]] | list[dict[str, Any]]:
        """
        Get LR scheduler instances from torch.optim.lr_scheduler or anyBrainer.models.schedulers.

        The returned LR scheduler is a dictionary with the instance and keys expected in Lightning's lr_scheduler_config.

        Special cases:
        - If multiple LR schedulers and optimizers are provided, return a list of lr_scheduler dictionaries.
        - If a single LR scheduler is provided with multiple optimizers, return a list of lr_scheduler
            dictionaries for each optimizer.

        Args:
            lr_scheduler_kwargs: dict[str, Any] - lr scheduler kwargs. Must contain "name" key.
            optimizer: optim.Optimizer | list[optim.Optimizer] - optimizer or list of optimizers.
            cls_only: bool - whether to return only the class or the instance.

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

            if cls_only:
                return lr_scheduler_cls

        except AttributeError:
            try:
                lr_scheduler_cls = getattr(lr_schedulers, lr_scheduler_dict["name"])
                if cls_only:
                    return lr_scheduler_cls

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

    @classmethod
    def get_param_scheduler_instances_from_kwargs(
        cls,
        other_schedulers: list[dict[str, Any]],
        cls_only: bool = False,
    ) -> (tuple[list[param_schedulers.ParameterScheduler], list[param_schedulers.ParameterScheduler]] | 
          tuple[list[type[param_schedulers.ParameterScheduler]], list[type[param_schedulers.ParameterScheduler]]]):
        """
        Get any other custom schedulers from anyBrainer.models.schedulers.
        
        Groups the schedulers into step and epoch schedulers, the values of which are
        extracted using built-in Lightning hooks.
        Always assuming list of dicts.

        Args:
            other_schedulers: list[dict[str, Any]] - list of scheduler kwargs.
            cls_only: bool - whether to return only the class or the instance.

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
            scheduler_name = scheduler_kwargs.pop("name")
            scheduler_interval = scheduler_kwargs.pop("interval")
            
            try:
                scheduler_cls = getattr(param_schedulers, scheduler_name)
            except AttributeError:
                msg = f"Scheduler '{scheduler_name}' not found in anyBrainer.models.schedulers."
                logger.error(msg)
                raise ValueError(msg)   
            
            # Handle improper initialization args
            if not cls_only:
                try:
                    scheduler = scheduler_cls(**scheduler_kwargs)
                except Exception as e:
                    msg = f"Error initializing scheduler '{scheduler_name}': {e}"
                    logger.exception(msg)
                    raise
            else:
                scheduler = scheduler_cls
            
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
        cls_only: bool = False,
    ) -> type[nn.Module] | nn.Module | list[type[nn.Module]] | list[nn.Module]:
        """
        Get loss function instances from torch.nn or anyBrainer.models.losses.

        Special cases:
        - If multiple loss functions are provided, return a list of loss function instances.

        Args:
            loss_fn_kwargs: dict[str, Any] | list[dict[str, Any]] - loss fn kwargs.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
        - ValueError: If loss fn name is not found in loss_fn_kwargs.
        - ValueError: If loss fn name is not found in torch.nn or anyBrainer.models.losses.
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
            if cls_only:
                return loss_fn_cls

        except AttributeError:
            try:
                loss_fn_cls = getattr(losses, loss_fn_name)
                if cls_only:
                    return loss_fn_cls

            except AttributeError:  
                msg = (f"Loss fn '{loss_fn_name}' not found in torch.nn or "
                        f"anyBrainer.models.losses.")
                logger.error(msg)
                raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            loss_fn = loss_fn_cls(**loss_fn_kwargs)
        except Exception as e:
            msg = f"Error initializing loss fn '{loss_fn_name}': {e}"
            logger.exception(msg)
            raise

        return loss_fn

    @classmethod
    def get_pl_callback_instances_from_kwargs(
        cls,
        callback_kwargs: list[dict[str, Any]],
        cls_only: bool = False,
    ) -> list[pl.Callback] | list[type[pl.Callback]]:
        """
        Get callback instances from anyBrainer.engines.callbacks or pl.callbacks. 

        Always assuming list of dicts.

        Args:
            callback_kwargs: list[dict[str, Any]] - list of callback kwargs.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
        - ValueError: If callback name is not found in callback_kwargs.
        - ValueError: If requested callback is not found in anyBrainer.engines.callbacks or pl.callbacks.
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
                callback_cls = getattr(callbacks, callback_name)
            except AttributeError:
                try:
                    callback_cls = getattr(pl_callbacks, callback_name)
                except AttributeError:
                    msg = (f"Callback '{callback_name}' not found in anyBrainer.engines.callbacks "
                        "or pl.callbacks.")
                    logger.error(msg)
                    raise ValueError(msg)
            
            if cls_only:
                callbacks_list.append(callback_cls)
            else:
                try: # Handle improper initialization args
                    callbacks_list.append(callback_cls(**_callback_kwargs))
                except Exception as e:
                    msg = f"Error initializing callback '{callback_name}': {e}"
                    logger.exception(msg)
                    raise
        
        return callbacks_list

    @classmethod
    def get_transformslist_from_kwargs(
        cls,
        transform_kwargs: dict[str, Any],
        cls_only: bool = False,
    ) -> list[Callable]: 
        """
        Get list of transforms from anyBrainer.transforms, using a transform builder fn. 

        Args:
            transform_kwargs: dict[str, Any] - transform kwargs. Must contain "name" key.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
        - ValueError: If transform name is not present in transform_kwargs.
        - ValueError: If requested transform is not found in anyBrainer.transforms.
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
            get_transforms_fn = getattr(transforms, transforms_name)
            if cls_only:
                return get_transforms_fn

        except AttributeError:
            msg = f"Transform builder name '{transforms_name}' not found in anyBrainer.transforms."
            logger.error(msg)
            raise ValueError(msg)
        
        # Handle improper initialization args
        try:
            transformslist = get_transforms_fn(**transform_kwargs)
        except Exception as e:
            msg = f"Error retrieving transormslist '{transforms_name}': {e}"
            logger.exception(msg)
            raise

        return transformslist


class ModuleFactory:
    @classmethod
    def get_pl_module_instance_from_kwargs(
        cls,
        pl_module_kwargs: dict[str, Any],
        cls_only: bool = False,
    ) -> pl.LightningModule:
        """
        Get pl module instance from anyBrainer.engines.models.

        Args:
            pl_module_kwargs: dict[str, Any] - pl module kwargs. Must contain "name" key.
            cls_only: bool - whether to return only the class or the instance.

        TODO: add support for registry from pytorch_lightning.

        Raises:
        - ValueError: If PL module name is not found in pl_module_kwargs.
        - ValueError: If requested PL module is not found in pytorch_lightning.
        - Exception: If error occurs during PL module initialization.
        """
        if "name" not in pl_module_kwargs:
            msg = "PL module name not found in pl_module_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        pl_module_kwargs = pl_module_kwargs.copy()
        pl_module_name = pl_module_kwargs.pop("name")

        try:
            pl_module_cls = getattr(models, pl_module_name)
            if cls_only:
                return pl_module_cls

        except AttributeError:
            msg = f"PL module '{pl_module_name}' not found in pytorch_lightning."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            pl_module = pl_module_cls(**pl_module_kwargs)
        except Exception as e:
            msg = f"Error initializing PL module '{pl_module_name}': {e}"
            logger.exception(msg)
            raise
        
        return pl_module
    
    @classmethod
    def get_pl_datamodule_instance_from_kwargs(
        cls,
        pl_datamodule_kwargs: dict[str, Any],
        cls_only: bool = False,
    ) -> pl.LightningModule:
        """
        Get pl module instance from anyBrainer.engines.models.

        Args:
            pl_datamodule_kwargs: dict[str, Any] - pl datamodule kwargs. Must contain "name" key.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
        - ValueError: If PL datamodule name is not found in pl_datamodule_kwargs.
        - ValueError: If requested PL datamodule is not found in anyBrainer.data.datamodule.
        - Exception: If error occurs during PL datamodule initialization.
        """
        if "name" not in pl_datamodule_kwargs:
            msg = "PL datamodule name not found in pl_datamodule_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        pl_datamodule_kwargs = pl_datamodule_kwargs.copy()
        pl_datamodule_name = pl_datamodule_kwargs.pop("name")

        try:
            pl_datamodule_cls = getattr(datamodules, pl_datamodule_name)
            if cls_only:
                return pl_datamodule_cls

        except AttributeError:
            msg = f"PL datamodule '{pl_datamodule_name}' not found in anyBrainer.data.datamodule."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            pl_datamodule = pl_datamodule_cls(**pl_datamodule_kwargs)
        except Exception as e:
            msg = f"Error initializing PL datamodule '{pl_datamodule_name}': {e}"
            logger.exception(msg)
            raise
        
        return pl_datamodule
    
    @classmethod
    def get_logging_manager_instance_from_kwargs(
        cls,
        logging_manager_kwargs: dict[str, Any],
        cls_only: bool = False,
    ) -> logging_managers.LoggingManager | type[logging_managers.LoggingManager]:
        """
        Get logging manager instance from anyBrainer.log.logging_manager.

        Args:
            logging_manager_kwargs: dict[str, Any] - logging manager kwargs.
            cls_only: bool - whether to return only the class or the instance.

        Raises:
        - ValueError: If logging manager name is not found in logging_manager_kwargs.
        - ValueError: If requested logging manager is not found in anyBrainer.log.logging_manager.
        - Exception: If error occurs during logging manager initialization.
        """
        if "name" not in logging_manager_kwargs:
            msg = "Logging manager name not found in logging_manager_kwargs."
            logger.error(msg)
            raise ValueError(msg)
        
        logging_manager_kwargs = logging_manager_kwargs.copy()
        logging_manager_name = logging_manager_kwargs.pop("name")

        try:
            logging_manager_cls = getattr(logging_managers, logging_manager_name)
            if cls_only:
                return logging_manager_cls

        except AttributeError:
            msg = f"Logging manager '{logging_manager_name}' not found in anyBrainer.log.logging_manager."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            logging_manager = logging_manager_cls(**logging_manager_kwargs)
        except Exception as e:
            msg = f"Error initializing logging manager '{logging_manager_name}': {e}"
            logger.exception(msg)
            raise
        
        return logging_manager