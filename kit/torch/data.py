from __future__ import annotations
from abc import abstractmethod
from typing import Iterator, Sequence, Sized

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataset import Subset, random_split

from kit import implements

__all__ = ["prop_random_split", "InfSequentialBatchSampler", "StratifiedSampler"]


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


class InfBatchSampler(Sampler[Sequence[int]]):
    @implements(Sampler)
    @abstractmethod
    def __iter__(self) -> Iterator[list[int]]:
        ...

    def __len__(self) -> None:
        """The number of samples drawn.
        Since such samplers are inherently non-terminating, their length is undefined.
        However, __len__ still needs to be defined for downstream compatibility
        (e.g. with PyTorch Lightning) and for this it suffices to simply return None.
        """
        return None


class InfSequentialBatchSampler(InfBatchSampler):
    r"""Infinitely samples elements sequentially, always in the same order.
    This is useful for enabling iteration-based training.
    Note that unlike torch's SequentialSampler which is an ordinary sampler that yields independent sample indexes,
    this is a BatchSampler, requiring slightly different treatment when used with a DataLoader.

    Example:
        >>> batch_sampler = InfSequentialBatchSampler(data_source=train_data, batch_size=100, shuffle=True)
        >>> train_loader = DataLoader(train_data, batch_sampler=batch_sampler, shuffle=False, drop_last=False) # drop_last and shuffle need to be False
        >>> train_loader_iter = iter(train_loader)
        >>> for _ in range(train_iters):
        >>>     batch = next(train_loader_iter)

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

    @implements(InfBatchSampler)
    def __iter__(self) -> Iterator[list[int]]:
        batched_idxs_iter = iter(self.batch_indexes(self._generate_idx_seq()))
        # Iterate until some externally-defined stopping criterion is reached
        while True:
            batch_idxs = next(batched_idxs_iter, None)  # type: ignore
            if batch_idxs is None or (len(batch_idxs) < self.batch_size):
                new_idx_seq = self._generate_idx_seq()
                if batch_idxs is not None:
                    # Rather than dropping the last batch if it is incomplete or simply using it,
                    # incomplete as it may be, we take the alternative approach of concatenating the surplus
                    # batch to the beginning of the next generation of indexes
                    new_idx_seq = torch.cat([batch_idxs, new_idx_seq])
                batched_idxs_iter = iter(self.batch_indexes(new_idx_seq))
            else:
                yield batch_idxs.tolist()


class StratifiedSampler(InfBatchSampler):
    r"""Samples equal proportion of elements from ``[0,..,len(group_ids)-1]``.

    To drop certain groups, set their multiplier to 0.

    Args:
        group_ids: a sequence of group IDs, not necessarily contiguous.
        num_samples_per_group: number of samples to draw per group. Note that if a multiplier is > 1
            then effectively more samples will be drawn for that group.
        replacement: if ``True``, samples are drawn with replacement. If not, they are drawn without
            replacement, which means that when a sample index is drawn for a row, it cannot be drawn
            again for that row.
        multiplier: an optional dictionary that maps group IDs to multipliers. If a multiplier is
            greater than 1, the corresponding group will be sampled at twice the rate as the other
            groups. If a multiplier is 0, the group will be skipped.

    Example:
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 2], 10, replacement=True))
        [3, 5, 6, 3, 5, 6, 0, 5, 6]
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 2], 10, replacement=True, multiplier={2: 2}))
        [3, 4, 6, 6, 3, 5, 6, 6, 1, 5, 6, 6]
        >>> list(StratifiedSampler([0, 0, 0, 0, 1, 1, 1, 2, 2], 7, replacement=False))
        [2, 6, 7, 0, 5, 8]
    """

    def __init__(
        self,
        group_ids: Sequence[int],
        num_samples_per_group: int,
        replacement: bool = True,
        multipliers: dict[int, int] | None = None,
    ) -> None:
        if (
            not isinstance(num_samples_per_group, int)
            or isinstance(num_samples_per_group, bool)
            or num_samples_per_group <= 0
        ):
            raise ValueError(
                f"num_samples_per_group should be a positive integer; got {num_samples_per_group}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )
        self.num_samples_per_group = num_samples_per_group
        self.replacement = replacement
        multipliers_ = {} if multipliers is None else multipliers

        group_ids_t = torch.as_tensor(group_ids, dtype=torch.int64)
        # find all unique IDs
        groups: list[int] = group_ids_t.unique().tolist()

        # get the indexes for each group separately and compute the effective number of groups
        groupwise_idxs: list[tuple[Tensor, int]] = []
        num_groups_effective = 0
        for group in groups:
            # Idxs needs to be 1 dimensional
            idxs = (group_ids_t == group).nonzero(as_tuple=False).view(-1)
            multiplier = multipliers_.get(group, 1)
            assert isinstance(multiplier, int) and multiplier >= 0, "multiplier has to be >= 0"
            groupwise_idxs.append((idxs, multiplier))
            num_groups_effective += multiplier

            if not replacement and len(idxs) < num_samples_per_group * multiplier:
                raise ValueError(
                    f"Not enough samples in group {group} to sample {num_samples_per_group}."
                )

        self.groupwise_idxs = groupwise_idxs
        self.num_groups_effective = num_groups_effective

    @implements(InfBatchSampler)
    def __iter__(self) -> Iterator[list[int]]:
        # loop over the groups and sample from each group separately
        while True:
            sampled_idxs: list[Tensor] = []
            for group_idx, multiplier in self.groupwise_idxs:
                if self.replacement:
                    for _ in range(multiplier):
                        # sampling with replacement:
                        # just sample enough random numbers to fill the quota
                        idx_of_idx = torch.randint(
                            low=0, high=len(group_idx), size=(self.num_samples_per_group,)
                        )
                        sampled_idxs.append(group_idx[idx_of_idx])
                else:
                    # sampling without replacement:
                    # first shuffle the indexes and then take as many as we need
                    shuffled_idx = group_idx[torch.randperm(len(group_idx))]
                    # all elements in `sampled_idx` have to have the same size,
                    # so we split the tensor in equal-sized parts and then take as many as we need
                    chunks = torch.split(shuffled_idx, self.num_samples_per_group)
                    sampled_idxs += list(chunks[:multiplier])

            yield torch.cat(sampled_idxs, dim=0).tolist()
