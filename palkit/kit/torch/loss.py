from __future__ import annotations
from enum import Enum, auto
from functools import partial
from typing import Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from kit import parsable, str_to_enum

__all__ = ["CrossEntropyLoss", "ReductionType"]


class ReductionType(Enum):
    mean = auto()
    none = auto()
    sum = auto()
    batch_mean = auto()


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
        if reduction is not None:
            if isinstance(reduction, str):
                reduction = str_to_enum(str_=reduction, enum=ReductionType)
        else:
            reduction = self.reduction

        if input.ndim == 1 or input.size(1) == 1:  # Binary classification
            target = target.view_as(input)
            if not target.is_floating_point():
                target = target.float()
            loss_fn = F.binary_cross_entropy_with_logits
        else:  # Multiclass classification
            target = target.flatten()
            if target.dtype != torch.long:
                target = target.long()
            loss_fn = partial(F.cross_entropy, ignore_index=self.ignore_index)
        losses = loss_fn(
            input=input,
            target=target,
            weight=self.weight,
            reduction="none",
        )
        if instance_weight is not None:
            _weight = instance_weight.flatten()
            losses *= _weight
        if self._reduction is ReductionType.mean:
            return losses.mean()
        elif self._reduction is ReductionType.batch_mean:
            return losses.mean(0)
        elif self._reduction is ReductionType.sum:
            return losses.sum()
        else:
            return losses
