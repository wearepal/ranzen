import pytest
import torch
from torch import Tensor
from torch.optim import AdamW

from ranzen.torch.optimizers import LAMB, SAM


@pytest.mark.parametrize("debias", [True, False])
def test_lamb(debias: bool):
    params = torch.randn(10, requires_grad=True)
    optimizer = LAMB(params=[params], debias=debias)
    old_params = params.data.clone()
    for _ in range(2):
        loss = params.norm()
        loss.backward()
        optimizer.step()
    assert not torch.allclose(old_params.data, params.data)


@pytest.mark.parametrize("adaptive", [True, False])
def test_sam(adaptive: bool):
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
