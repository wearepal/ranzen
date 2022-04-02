from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["batched_randint"]


def batched_randint(high: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
    r"""Batched version of :function:`torch.randint`.

    Randomly samples an integer from the domain :math:`[0, h_i]` for each sample :math:`h_i \in high`.

    :function:`torch.randint` requires ``high`` to be an integer and thus  prohibits having
    different samples within a batch having sampling domains, something which is necessary in order 
    to vectorise, for example, sampling from groups of different sizes or sampling objects with 
    different offsets. This function addresses this limitation using inverse transform sampling.

    :param high: A batch of tensors encoding the maximum integer value the corresponding random
        samples may take.

    :param generator: Pseudo-random-number generator to use for sampling.

    :returns: A tensor of randomly of the same shape as ``high``.
    """
    step_sizes = high.reciprocal()
    u = (
        torch.rand(size=high.size(), device=high.device, generator=generator) * (1 + step_sizes)
        - step_sizes / 2
    )
    return (u.clamp(min=0, max=1) * (high - 1)).round().long()
