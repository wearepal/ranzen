from typing import Callable
from typing_extensions import TypeAlias

from torch import Tensor

__all__ = ["LossClosure"]

LossClosure: TypeAlias = Callable[..., Tensor]
