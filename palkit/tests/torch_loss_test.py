import pytest
import torch
from typing_extensions import Final

from kit.torch import CrossEntropyLoss, ReductionType

BATCH_SIZE: Final[int] = 3


@pytest.mark.parametrize("out_dim", [0, 1, 3])
@pytest.mark.parametrize("dtype", ["long", "float"])
@pytest.mark.parametrize("reduction_type", list(ReductionType))
def test_xent(out_dim: int, dtype: str) -> None:
    target = torch.randint(0, max(out_dim, 2), (BATCH_SIZE, 1), dtype=getattr(torch, dtype))
    pred = torch.randn(BATCH_SIZE, out_dim)
    loss_fn = CrossEntropyLoss()
    loss_fn(input=pred, target=target)
    loss_fn(input=pred, target=target.squeeze())
    if out_dim == 1:
        loss_fn(input=pred.squeeze(), target=target)
