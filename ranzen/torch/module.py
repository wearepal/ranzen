from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from typing_extensions import Self, final

__all__ = ["DcModule"]


@dataclass(unsafe_hash=True)
class DcModule(nn.Module):
    @final
    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj
