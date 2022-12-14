from __future__ import annotations
import operator
from typing import Final, Literal, cast

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from ranzen.torch.transforms import MixUpMode, RandomMixUp

BATCH_SIZE: Final[int] = 20
NUM_CLASSES: Final[int] = 5


@pytest.mark.parametrize("lambda_dist", ["beta", "uniform", "bernoulli"])
@pytest.mark.parametrize("mode", list(MixUpMode))
@pytest.mark.parametrize("p", [0.8, 1])
@pytest.mark.parametrize("one_hot", [True, False])
@pytest.mark.parametrize("num_classes", [3, None])
@pytest.mark.parametrize("num_groups", [2, None])
@pytest.mark.parametrize("input_shape", [(17,), (3, 9, 7)])
@pytest.mark.parametrize("featurewise", [True, False])
@pytest.mark.parametrize("cross_group", [True, False])
@pytest.mark.parametrize("edges", [True, False])
def test_mixup(
    lambda_dist: Literal["beta", "uniform", "bernoulli"],
    mode: MixUpMode,
    p: float,
    one_hot: bool,
    num_classes: int | None,
    num_groups: int | None,
    input_shape: tuple[int, ...],
    featurewise: bool,
    cross_group: bool,
    edges: bool,
) -> None:
    generator = torch.Generator().manual_seed(47)
    inputs = torch.randn(BATCH_SIZE, *input_shape, generator=generator)
    if num_classes is None:
        targets = None
    else:
        targets = torch.randint(
            low=0, high=num_classes, size=(BATCH_SIZE,), dtype=torch.long, generator=generator
        )
        if one_hot:
            targets = cast(Tensor, F.one_hot(targets, num_classes=num_classes))
    if num_groups is None:
        groups_or_edges = None
    else:
        groups_or_edges = torch.randint(
            low=0, high=num_groups, size=(BATCH_SIZE,), dtype=torch.long
        )
        if edges:
            comp = operator.ne if cross_group else operator.eq
            groups_or_edges = comp(groups_or_edges.unsqueeze(1), groups_or_edges)

    kwargs = dict(num_classes=num_classes, mode=mode, p=p, generator=generator)
    if lambda_dist != "bernoulli":
        kwargs["featurewise"] = featurewise
    transform = cast(
        RandomMixUp,
        getattr(RandomMixUp, f"with_{lambda_dist}_dist")(**kwargs),
    )
    res = transform(
        inputs=inputs,
        targets=targets,
        groups_or_edges=groups_or_edges,
        cross_group=cross_group,
    )
    if isinstance(res, tuple):
        assert targets is not None
        mixed_inputs = res.inputs
        mixed_targets = res.targets
        assert len(mixed_inputs) == len(mixed_targets) == BATCH_SIZE
        assert mixed_inputs.shape == inputs.shape
        assert mixed_targets.size(1) == num_classes

        # Check that setting rows to False in the connectivity matrix excludes samples.
        if (groups_or_edges is not None) and edges and (p == 1.0):
            groups_or_edges_t = groups_or_edges.clone()
            groups_or_edges_t[[0, -1]] = False
            inputs_mu = transform(
                inputs=inputs,
                targets=targets,
                groups_or_edges=groups_or_edges_t,
                num_classes=num_classes,
            ).inputs
            assert torch.allclose(inputs_mu[[0, -1]], inputs[[0, -1]])

        if not one_hot:
            with pytest.raises(RuntimeError):
                transform.num_classes = None
                transform(
                    inputs=inputs,
                    targets=targets,
                    groups_or_edges=groups_or_edges,
                    cross_group=cross_group,
                )
    else:
        assert targets is None
        assert res.shape == inputs.shape
