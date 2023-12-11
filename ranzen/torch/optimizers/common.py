from collections.abc import Callable
from typing import TypeAlias

from torch import Tensor

__all__ = ["LossClosure"]

LossClosure: TypeAlias = Callable[..., Tensor]
