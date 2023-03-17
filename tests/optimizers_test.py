from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.optim import AdamW

from ranzen.torch.optimizers import Adafactor, LAMB, SAM


@pytest.mark.parametrize("debias", [True, False])
def test_lamb(debias: bool) -> None:
    params = torch.randn(10, requires_grad=True)
    optimizer = LAMB(params=[params], debias=debias)
    old_params = params.data.clone()
    for _ in range(2):
        loss = params.norm()
        loss.backward()
        optimizer.step()
    assert not torch.allclose(old_params.data, params.data)


@pytest.mark.parametrize("adaptive", [True, False])
def test_sam(adaptive: bool) -> None:
    params = torch.randn(10, requires_grad=True)
    base_optimizer = AdamW([params])
    optimizer = SAM(base_optimizer=base_optimizer, adaptive=adaptive)

    def _closure() -> Tensor:
        return params.norm()

    old_params = params.data.clone()
    for _ in range(2):
        loss = _closure()
        loss.backward()
        optimizer.step(closure=_closure)
    assert not torch.allclose(old_params.data, params.data)


@pytest.mark.parametrize("lr", [1.0, None])
@pytest.mark.parametrize("beta1", [0.98, None])
@pytest.mark.parametrize("multiply_by_parameter_scale", [True, False])
@pytest.mark.parametrize("warmup_init", [True, False])
def test_adafactor(
    lr: float | None, beta1: float | None, multiply_by_parameter_scale: bool, warmup_init: bool
):
    params = torch.randn(10, requires_grad=True)
    optimizer = Adafactor(
        params=[params],
        lr=lr,
        beta1=beta1,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        warmup_init=warmup_init,
    )
    for _ in range(2):
        loss = params.norm()
        loss.backward()
        optimizer.step()
