from __future__ import annotations

import pytest
import torch
from typing_extensions import Final

from kit.torch.transforms import BernoulliMixUp, BetaMixUp, MixUpMode, UniformMixUp

BATCH_SIZE: Final[int] = 7
NUM_CLASSES: Final[int] = 3


@pytest.mark.parametrize("mixup_cls", [BernoulliMixUp, BetaMixUp, UniformMixUp])
@pytest.mark.parametrize("mode", list(MixUpMode))
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("num_classes", [3, None])
@pytest.mark.parametrize("input_shape", [(7,), (3, 5, 5)])
def test_mixup(
    mixup_cls: type[BernoulliMixUp] | type[BetaMixUp] | type[UniformMixUp],  # type: ignore
    mode: MixUpMode,  # type: ignore
    p: float,
    num_classes: int | None,
    input_shape: tuple[int, ...],
) -> None:
    inputs = torch.randn(BATCH_SIZE, *input_shape)
    if num_classes is None:
        targets = None
    else:
        targets = torch.randint(low=0, high=num_classes, size=(BATCH_SIZE,))

    transform = mixup_cls(num_classes=num_classes, mode=mode, p=p)

    res = transform(inputs=inputs, targets=targets)
    if isinstance(res, tuple):
        assert targets is not None
        mixed_inputs = res.inputs
        mixed_targets = res.targets
        assert len(mixed_inputs) == len(mixed_targets) == BATCH_SIZE
        assert mixed_inputs.shape == inputs.shape
        assert mixed_targets.size(1) == num_classes

        with pytest.raises(ValueError):
            transform.num_classes = None
            transform(inputs=inputs, targets=targets)
    else:
        assert targets is None
        assert res.shape == inputs.shape
