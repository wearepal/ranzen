from __future__ import annotations
from enum import Enum, auto
from functools import partial
from typing import Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ranzen import parsable, str_to_enum

__all__ = [
    "CrossEntropyLoss",
    "ReductionType",
    "cross_entropy_loss",
]


class ReductionType(Enum):
    """An enum for the type of reduction to apply to a batch of losses."""

    mean = auto()
    """compute the mean over all dimensions."""
    none = auto()
    """no reduction."""
    sum = auto()
    """compute the sum over all dimensions."""
    batch_mean = auto()
    """compute the mean over the batch (first) dimension, the sum over the remaining dimensions."""


def _reduce(losses: Tensor, reduction_type: ReductionType | str) -> Tensor:
    if isinstance(reduction_type, str):
        reduction_type = str_to_enum(str_=reduction_type, enum=ReductionType)
    if reduction_type is ReductionType.mean:
        return losses.mean()
    elif reduction_type is ReductionType.batch_mean:
        return losses.sum() / losses.size(0)
    elif reduction_type is ReductionType.sum:
        return losses.sum()
    elif reduction_type is ReductionType.none:
        return losses
    raise TypeError(
        f"Received invalid type '{type(reduction_type)}' for argument 'reduction_type'."
    )


def cross_entropy_loss(
    input: Tensor,
    *,
    target: Tensor,
    instance_weight: Tensor | None = None,
    reduction: ReductionType | str = ReductionType.mean,
    ignore_index: int = -100,
    weight: Tensor | None = None,
) -> Tensor:
    if isinstance(reduction, str):
        reduction = str_to_enum(str_=reduction, enum=ReductionType)
    if input.ndim == 1 or input.size(1) == 1:  # Binary classification
        target = target.view_as(input)
        if not target.is_floating_point():
            target = target.float()
        loss_fn = F.binary_cross_entropy_with_logits
    else:  # Multiclass classification
        target = target.flatten()
        if target.dtype != torch.long:
            target = target.long()
        loss_fn = partial(F.cross_entropy, ignore_index=ignore_index)
    losses = loss_fn(
        input=input,
        target=target,
        weight=weight,
        reduction="none",
    )
    if instance_weight is not None:
        losses *= instance_weight.view_as(losses)
    return _reduce(losses=losses, reduction_type=reduction)


class CrossEntropyLoss(nn.Module):
    weight: Tensor | None

    @parsable
    def __init__(
        self,
        *,
        class_weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: Union[ReductionType, str] = ReductionType.mean,
    ) -> None:
        super().__init__()
        if isinstance(reduction, str):
            reduction = str_to_enum(str_=reduction, enum=ReductionType)
        self.register_buffer("weight", class_weight)
        self.ignore_index = ignore_index
        self._reduction = reduction

    @property
    def reduction(self) -> ReductionType:
        return self._reduction

    @reduction.setter
    def reduction(self, value: ReductionType | str) -> None:  # type: ignore
        if isinstance(value, str):
            value = str_to_enum(str_=value, enum=ReductionType)
        self._reduction = value

    def forward(
        self,
        input: Tensor,
        *,
        target: Tensor,
        instance_weight: Tensor | None = None,
        reduction: ReductionType | str | None = None,
    ) -> Tensor:
        reduction = self.reduction if reduction is None else reduction
        return cross_entropy_loss(
            input=input,
            target=target,
            instance_weight=instance_weight,
            reduction=reduction,
            weight=self.weight,
        )
