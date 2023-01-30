import pytest
import torch
from torch import optim

from ranzen.torch.schedulers import LinearWarmupLR


def test_linear_warmup_lr() -> None:
    params = (torch.randn(1, 1, requires_grad=True),)
    base_lr = 1.0
    lr_start = 1.0e-1
    optimizer = optim.SGD(params, lr=base_lr)
    scheduler = LinearWarmupLR(optimizer=optimizer, lr_start=lr_start, warmup_iters=1)
    for group in optimizer.param_groups:
        assert group["lr"] == lr_start

    def _step():
        optimizer.step()
        scheduler.step()

    _step()
    for group in optimizer.param_groups:
        assert group["lr"] == base_lr

    optimizer = optim.AdamW(params, lr=base_lr)
    scheduler = LinearWarmupLR(optimizer=optimizer, lr_start=lr_start, warmup_iters=0)
    for group in optimizer.param_groups:
        assert group["lr"] == base_lr

    _step()
    for group in optimizer.param_groups:
        assert group["lr"] == base_lr

    optimizer = optim.AdamW(params, lr=base_lr)
    scheduler = LinearWarmupLR(optimizer=optimizer, lr_start=lr_start, warmup_iters=2)
    _step()
    expected_lr_after_one_step = lr_start + 0.5 * (base_lr - lr_start)
    for group in optimizer.param_groups:
        assert group["lr"] == expected_lr_after_one_step

    with pytest.raises(AttributeError):
        scheduler = LinearWarmupLR(optimizer=optimizer, lr_start=1.0e-1, warmup_iters=-1)
