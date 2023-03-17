from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, field
import math
from typing import Generic, List, Optional, TypeVar, Union, overload
from typing_extensions import override

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

__all__ = [
    "CosineLRWithLinearWarmup",
    "CosineWarmup",
    "ExponentialWarmup",
    "LinearWarmup",
    "LinearWarmupLR",
    "Scheduler",
    "WarmupScheduler",
]


class LinearWarmupLR(_LRScheduler):
    """
    Applies a linear warmup schedule to the learning rate, increasing/decreasing it by a fixed
    step-size from lr_start to the base value over the specified number of warmup steps.
    """

    last_epoch: int
    base_lrs: List[float]
    optimizer: Optimizer

    def __init__(
        self, optimizer: Optimizer, *, warmup_iters: int, lr_start: float = 0, last_epoch: int = -1
    ) -> None:
        if warmup_iters < 0:
            raise AttributeError("'warmup_iters' must be non-negative.")
        self.lr_start = lr_start
        self.warmup_iters = warmup_iters
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    @override
    def get_lr(self) -> list[float]:  # type: ignore
        """
        Get the learning rate of each parameter group.

        :returns: The learning rate for each parameter group in the optimizer.
        """
        if self.last_epoch > self.warmup_iters or (self.warmup_iters == 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            self.lr_start + self._get_step_size(base_lr) * self.last_epoch
            for base_lr in self.base_lrs
        ]

    def _get_step_size(self, base_lr: float) -> float:
        return (base_lr - self.lr_start) / self.warmup_iters

    def _get_closed_form_lr(self) -> list[float]:
        return [
            self.lr_start
            + (base_lr - self.lr_start)
            * min(self.last_epoch, self.warmup_iters)
            / self.warmup_iters
            for base_lr in self.base_lrs
        ]


class CosineLRWithLinearWarmup(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule between
    lr_start and base_lr followed by a cosine annealing schedule between base_lr and lr_min.
    """

    last_epoch: int
    base_lrs: List[float]
    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_iters: Union[int, float],
        lr_start: float = 0.0,
        total_iters: int,
        lr_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        :param optimizer: Optimizer whose parameter groups are to be scheduled.
        :param warmup_iters: Maximum number of iterations for linear warmup.
            Float values are interpreted as a fraction of ``total_iters``.

        :param total_iters: Total number of iterations.
        :param lr_start: Learning rate at the beginning of linear warmup.
        :param lr_min: Minimum learning rate permitted with cosine annealing.
        :param last_epoch: The index of the last epoch.

        :raises AttributeError: If warmup_iters is a float and not in the range [0, 1].
        """
        self.total_iters = total_iters
        self.lr_start = lr_start
        self.lr_min = lr_min
        if isinstance(warmup_iters, float):
            if not (0 <= warmup_iters <= 1):
                raise AttributeError(
                    "If 'warmup_iters' is a float, it must be in the range [0, 1]."
                )
            warmup_iters = round(warmup_iters * total_iters)
        elif warmup_iters < 0:
            raise AttributeError("If 'warmup_iters' is an integer, it must be non-negative.")
        self.warmup_iters = warmup_iters
        self._scheduler: Union[LinearWarmupLR, CosineAnnealingLR] = LinearWarmupLR(
            optimizer=optimizer, warmup_iters=warmup_iters, lr_start=lr_start
        )
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    @property
    def scheduler(self) -> Union[LinearWarmupLR, CosineAnnealingLR]:
        """
        The scheduler currently in use, as determined by the curren step. If the current stepe
        exceeds the number of warmup iterations, then the cosine scheduler will be returned, else
        the learning-warmup scheduler will be.

        :returns: The learning-rate scheduler currently in use.
        """
        # Switch to cosine-scheduling once the warmup-period is complete.
        if self.last_epoch == self.warmup_iters:
            self._scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.total_iters - self.warmup_iters + 1,
                eta_min=self.lr_min,
            )
        return self._scheduler

    @override
    def get_lr(self) -> list[float]:  # type: ignore
        """
        Get the learning rate of each parameter group.

        :returns: The learning rate for each parameter group in the optimizer.
        """
        return self.scheduler.get_lr()  # type: ignore

    @override
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Update the learning rates using the currently-used scheduler.
        """
        self.last_epoch = (self.last_epoch + 1) if epoch is None else epoch
        return self.scheduler.step(epoch)


T = TypeVar("T", Tensor, float)


@dataclass(unsafe_hash=True)
class Scheduler(Generic[T]):
    start_val: T
    val: T = field(init=False)

    def __post_init__(self) -> None:
        self.val = self.start_val

    @abstractmethod
    def _update(self, value: T) -> T:
        ...

    @torch.no_grad()
    def step(self) -> None:
        """Update the scheduled value."""
        self.val = self._update(self.val)

    @overload
    def __add__(self: "Scheduler[Tensor]", other: float) -> Tensor:
        ...

    @overload
    def __add__(self: "Scheduler[Tensor]", other: Tensor) -> Tensor:
        ...

    @overload
    def __add__(self: "Scheduler[float]", other: float) -> float:
        ...

    @overload
    def __add__(self: "Scheduler[float]", other: Tensor) -> Tensor:
        ...

    def __add__(self, other: Union[Tensor, float]) -> Union[Tensor, float]:
        return other + self.val

    @overload
    def __mul__(self: "Scheduler[Tensor]", other: float) -> Tensor:
        ...

    @overload
    def __mul__(self: "Scheduler[Tensor]", other: Tensor) -> Tensor:
        ...

    @overload
    def __mul__(self: "Scheduler[float]", other: float) -> float:
        ...

    @overload
    def __mul__(self: "Scheduler[float]", other: Tensor) -> Tensor:
        ...

    def __mul__(self, other: Union[Tensor, float]) -> Union[Tensor, float]:
        return other * self.val

    @overload
    def __imul__(self: "Scheduler[Tensor]", other: float) -> Tensor:
        ...

    @overload
    def __imul__(self: "Scheduler[Tensor]", other: Tensor) -> Tensor:
        ...

    @overload
    def __imul__(self: "Scheduler[float]", other: float) -> float:
        ...

    @overload
    def __imul__(self: "Scheduler[float]", other: Tensor) -> Tensor:
        ...

    def __imul__(self, other: Union[Tensor, float]) -> Union[Tensor, float]:
        return other * self.val

    @overload
    def __rmul__(self: "Scheduler[Tensor]", other: float) -> Tensor:
        ...

    @overload
    def __rmul__(self: "Scheduler[Tensor]", other: Tensor) -> Tensor:
        ...

    @overload
    def __rmul__(self: "Scheduler[float]", other: float) -> float:
        ...

    @overload
    def __rmul__(self: "Scheduler[float]", other: Tensor) -> Tensor:
        ...

    def __rmul__(self, other: Union[Tensor, float]) -> Union[Tensor, float]:
        return other * self.val


@dataclass(unsafe_hash=True)
class WarmupScheduler(Scheduler[T]):
    end_val: T
    warmup_steps: int
    _curr_step: int = field(init=False)

    @override
    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise AttributeError("'warmup_steps' must be a non-negative integer.")
        super().__post_init__()
        if self.warmup_steps == 0:
            self.val = self.end_val
        self._curr_step = 0

    @property
    def warmed_up(self) -> bool:
        return self._curr_step == self.warmup_steps

    @torch.no_grad()
    @override
    def step(self) -> None:
        if not self.warmed_up:
            super().step()
            self._curr_step += 1


@dataclass(unsafe_hash=True)
class LinearWarmup(WarmupScheduler[T]):
    step_size: T = field(init=False)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.warmup_steps == 0:
            self.step_size = 0  # type: ignore
        else:
            self.step_size = (self.end_val - self.start_val) / self.warmup_steps

    @override
    def _update(self, value: T) -> T:
        return value + self.step_size


@dataclass(unsafe_hash=True)
class ExponentialWarmup(WarmupScheduler[T]):
    end_val: T
    step_size: T = field(init=False)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.warmup_steps == 0:
            self.step_size = 0  # type: ignore
        else:
            self.step_size = (self.end_val / self.start_val) ** (  # pyright: ignore
                1 / self.warmup_steps
            )

    @override
    def _update(self, value: T) -> T:
        return value * self.step_size


@dataclass(unsafe_hash=True)
class CosineWarmup(WarmupScheduler[T]):
    end_val: T
    _coeff: T = field(init=False)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        self._coeff = 0.5 * (self.end_val - self.start_val)

    @override
    def _update(self, value: T) -> T:
        if self.warmup_steps == 0:
            return value
        return (
            self._coeff * (1 - math.cos(math.pi * (self._curr_step + 1) / self.warmup_steps))
            + self.start_val
        )
