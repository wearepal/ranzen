from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset, random_split

__all__ = ["prop_random_split"]


def prop_random_split(
    dataset: Dataset, props: Sequence[float], seed: int | None = None
) -> list[Subset]:
    """Splits a dataset based on proportions rather than on absolute sizes."""
    if not hasattr(dataset, "__len__"):
        raise ValueError(
            "Split proportions can only be computed for datasets with __len__ defined."
        )
    len_ = len(dataset)  # type: ignore
    sum_ = np.sum(props)  # type: ignore
    if (sum_ > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    return random_split(dataset, section_sizes, generator=generator)
