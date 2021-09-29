from __future__ import annotations
from typing import cast

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import Final, Literal

from mantra.torch.transforms import MixUpMode, RandomMixUp

BATCH_SIZE: Final[int] = 20
NUM_CLASSES: Final[int] = 5


@pytest.mark.parametrize("lambda_dist", ["beta", "uniform", "bernoulli"])
@pytest.mark.parametrize("mode", list(MixUpMode))
@pytest.mark.parametrize("p", [0.8, 1])
@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("num_classes", [3, None])
@pytest.mark.parametrize("num_groups", [2, None])
@pytest.mark.parametrize("input_shape", [(7,), (3, 5, 5)])
def test_mixup(
    lambda_dist: Literal["beta", "uniform", "bernoulli"],
    mode: MixUpMode,  # type: ignore
    p: float,
    one_hot: bool,
    num_classes: int | None,
    num_groups: int | None,
    input_shape: tuple[int, ...],
) -> None:
    inputs = torch.randn(BATCH_SIZE, *input_shape)
    if num_classes is None:
        targets = None
    else:
        targets = torch.randint(low=0, high=num_classes, size=(BATCH_SIZE,), dtype=torch.long)
        if one_hot:
            targets = cast(Tensor, F.one_hot(targets, num_classes=num_classes))
    if num_groups is None:
        group_labels = None
    else:
        group_labels = torch.randint(low=0, high=num_groups, size=(BATCH_SIZE,), dtype=torch.long)

    transform = cast(
        RandomMixUp,
        getattr(RandomMixUp, f"with_{lambda_dist}_distribution")(
            num_classes=num_classes, mode=mode, p=p
        ),
    )
    res = transform(inputs=inputs, targets=targets, group_labels=group_labels)
    if isinstance(res, tuple):
        assert targets is not None
        mixed_inputs = res.inputs
        mixed_targets = res.targets
        assert len(mixed_inputs) == len(mixed_targets) == BATCH_SIZE
        assert mixed_inputs.shape == inputs.shape
        assert mixed_targets.size(1) == num_classes

        if not one_hot:
            with pytest.raises(ValueError):
                transform.num_classes = None
                transform(inputs=inputs, targets=targets, group_labels=group_labels)
    else:
        assert targets is None
        assert res.shape == inputs.shape
