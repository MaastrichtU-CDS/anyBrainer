"""Unit tests for schedulers."""

import pytest

from anyBrainer.models.schedulers import StepwiseParameterScheduler

class TestSchedulers:
    def test_stepwise_parameter_scheduler(self):
        """Unit test for StepwiseParameterScheduler."""
        scheduler = StepwiseParameterScheduler(
            param_name="loss_weight",
            start_step=0,
            end_step=1000,
            start_value=0.0,
            end_value=1.0,
            mode="linear",
        )
        assert scheduler.get_value(0) == {"loss_weight": 0.0}
        assert scheduler.get_value(1000) == {"loss_weight": 1.0}
        assert scheduler.get_value(500) == {"loss_weight": 0.5}
        assert scheduler.get_value(2000) == {"loss_weight": 1.0}
        assert scheduler.get_value(10000) == {"loss_weight": 1.0}
        assert scheduler.get_value(-10) == {"loss_weight": 0.0}
    
    def test_stepwise_parameter_scheduler_step_function(self):
        """Unit test for StepwiseParameterScheduler performing a step function."""
        scheduler = StepwiseParameterScheduler(
            param_name="another_param",
            start_step=1000,
            end_step=1000,
            start_value=0.0,
            end_value=1.0,
            mode="linear",
        )
        assert scheduler.get_value(0) == {"another_param": 0.0}
        assert scheduler.get_value(500) == {"another_param": 0.0}
        assert scheduler.get_value(1001) == {"another_param": 1.0}
        assert scheduler.get_value(2000) == {"another_param": 1.0}