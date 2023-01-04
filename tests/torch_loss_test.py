import pytest
import torch
import torch.nn.functional as F
from typing_extensions import Final

from ranzen.torch import CrossEntropyLoss, ReductionType

BATCH_SIZE: Final[int] = 3


@pytest.mark.parametrize("out_dim", [1, 3])
@pytest.mark.parametrize("dtype", ["long", "float"])
@pytest.mark.parametrize("reduction_type", list(ReductionType))
@pytest.mark.parametrize("le", [True, False])
def test_xent(out_dim: int, dtype: str, reduction_type: ReductionType, le: bool) -> None:
    num_classes = max(out_dim, 2)
    if le:
        target = torch.randint(0, num_classes, (BATCH_SIZE, 1), dtype=getattr(torch, dtype))
    else:
        if dtype == "long":
            target_le = torch.randint(0, num_classes, (BATCH_SIZE,), dtype=getattr(torch, dtype))
            target = F.one_hot(target_le, num_classes=num_classes)
        else:
            target = torch.randn(BATCH_SIZE, num_classes).softmax(dim=1)

    pred = torch.randn(BATCH_SIZE, out_dim)
    iw = torch.randn(BATCH_SIZE)
    loss_fn = CrossEntropyLoss(reduction=reduction_type)
    loss_fn(input=pred, target=target, instance_weight=iw)
    loss_fn(input=pred, target=target.squeeze(), instance_weight=iw)
    loss_fn(input=pred.unsqueeze(-1), target=target.unsqueeze(-1), instance_weight=iw)
    if out_dim == 1:
        loss_fn(input=pred.squeeze(), target=target, instance_weight=iw)
