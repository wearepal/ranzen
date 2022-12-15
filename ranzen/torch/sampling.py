from __future__ import annotations
from typing import Sequence

import torch
from torch import Tensor

__all__ = ["batched_randint"]


def batched_randint(
    high: Tensor,
    *,
    size: int | Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Batched version of :func:`torch.randint`.

    Randomly samples an integer from the domain :math:`[0, h_i]` for each sample :math:`h_i \in high`.

    :func:`torch.randint` requires ``high`` to be an integer and thus  prohibits having
    different samples within a batch having sampling domains, something which is necessary in order
    to vectorise, for example, sampling from groups of different sizes or sampling objects with
    different offsets. This func addresses this limitation using inverse transform sampling.

    :param high: A batch of tensors encoding the maximum integer value the corresponding random
        samples may take.

    :param size: An integer or sequence of integers defining the shape of the output tensor for each
        upper-bound specified in ``high``. The overall size of the sampled tensor will be
        'size(``high``) + ``size``. If ``None`` then the output size is simply 'size(``high``)'.

    :param generator: Pseudo-random-number generator to use for sampling.

    :returns: A tensor of random-sampled integers upper-bounded by the values in ``high``.
    """
    total_size: torch.Size | list[int] = high.size()
    if size is not None:
        total_size = list(total_size)
        if isinstance(size, int):
            total_size.append(size)
        else:
            total_size.extend(size)

    step_sizes = high.reciprocal()
    u = (
        torch.rand(size=total_size, device=high.device, generator=generator) * (1 + step_sizes)
        - step_sizes / 2
    )
    return (u.clamp(min=0, max=1) * (high - 1)).round().long()
