from __future__ import annotations
from typing import Iterator, List, Sequence, Sized

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataset import Subset, random_split

from kit import implements

__all__ = ["prop_random_split", "InfSequentialSampler"]


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


class InfSequentialSampler(Sampler[List[int]]):
    r"""Infinitely samples elements sequentially, always in the same order.
    This is useful for enabling iteration-based training.

    Args:
        data_source (Sized): dataset to sample from
    """

    def __init__(self, data_source: Sized, batch_size: int, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        self._dataset_size = len(data_source)
        self.batch_size = min(batch_size, self._dataset_size)

    def _generate_idx_seq(self) -> Tensor:
        """Generate a random sequence of unique indexes."""
        if self.shuffle:
            return torch.arange(self._dataset_size)
        return torch.randperm(self._dataset_size)

    def batch_indexes(self, indexes: Tensor) -> Sequence[Tensor]:
        """Split the indexes into batches."""
        return indexes.split(self.batch_size)

    @implements(Sampler)
    def __iter__(self) -> Iterator[List[int]]:
        batched_idxs_iter = iter(self.batch_indexes(self._generate_idx_seq()))
        # Iterate until some externally-defined stopping criterion is reached
        while True:
            batch_idxs = next(batched_idxs_iter, None)  # type: ignore
            if batch_idxs is None or (len(batch_idxs) < self.batch_size):
                new_idx_seq = self._generate_idx_seq()
                if batch_idxs is not None:
                    # Rather than dropping the last batch if it is incomplete or simply using it,
                    # incomlpete as it may be, we take the alternative approach of concatenating the surplus
                    # batch to the beginning of the next generation of indexes
                    new_idx_seq = torch.cat([batch_idxs, new_idx_seq])
                batched_idxs_iter = iter(self.batch_indexes(new_idx_seq))
            else:
                yield batch_idxs.tolist()

    def __len__(self) -> None:
        """The number of samples drawn.
        Since the sampler is by design non-terminating, its length is undefined.
        However__len__ still needs to be defined for downstream compatibility
        (e.g. with PyTorch Lightning) and for this it suffices to simply return None.
        """
        return None
