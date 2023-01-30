from typing import Callable

from torch import Tensor
from typing_extensions import TypeAlias

__all__ = ["LossClosure"]

LossClosure: TypeAlias = Callable[..., Tensor]
