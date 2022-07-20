from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from typing_extensions import Self, final
import torch.nn as nn

__all__ = ["DcModule"]

@dataclass(unsafe_hash=True)
class DcModule(nn.Module):
    @final
    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj
