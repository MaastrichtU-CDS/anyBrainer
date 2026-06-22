"""Unit tests for schedulers."""

import pytest
import torch
from torch.optim import SGD

from anyBrainer.core.schedulers import StepwiseParameterScheduler
from anyBrainer.core.schedulers.lr_schedulers import PolyLRWithWarmup


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
        """Unit test for StepwiseParameterScheduler performing a step
        function."""
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


class TestPolyLRWithWarmup:
    @pytest.fixture
    def scheduler(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = SGD([param], lr=0.01)
        return PolyLRWithWarmup(
            optimizer,
            warmup_iters=10,
            total_iters=100,
            exponent=0.9,
        )

    def test_inactive_before_start(self, scheduler):
        scheduler.last_epoch = -1
        assert scheduler.get_lr() == [0.0]

    def test_warmup_ramps_to_base_lr(self, scheduler):
        scheduler.last_epoch = 0
        assert scheduler.get_lr()[0] == pytest.approx(0.01 * 1 / 10)
        scheduler.last_epoch = 9
        assert scheduler.get_lr()[0] == pytest.approx(0.01)

    def test_poly_decay_mid_training(self, scheduler):
        scheduler.last_epoch = 55
        # effective=55, warmup=10, decay_iters=90, progress=(55-10)/90
        progress = (55 - 10) / 90.0
        expected = 0.01 * (1.0 - progress) ** 0.9
        assert scheduler.get_lr()[0] == pytest.approx(expected)

    def test_poly_decay_reaches_zero(self, scheduler):
        scheduler.last_epoch = 100
        assert scheduler.get_lr()[0] == pytest.approx(0.0)
