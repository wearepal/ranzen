from __future__ import annotations
from abc import abstractmethod
from enum import Enum, auto
import math
from typing import Iterator, Sequence, Sized

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataset import Subset, random_split

from kit import implements
from kit.misc import str_to_enum

__all__ = ["prop_random_split", "SequentialBatchSampler", "StratifiedBatchSampler", "TrainingMode"]


def prop_random_split(
    dataset: Dataset, *, props: Sequence[float] | float, seed: int | None = None
) -> list[Subset]:
    """Splits a dataset based on proportions rather than on absolute sizes."""
    if not hasattr(dataset, "__len__"):
        raise ValueError(
            "Split proportions can only be computed for datasets with __len__ defined."
        )
    if isinstance(props, float):
        props = [props]
    len_ = len(dataset)  # type: ignore
    sum_ = np.sum(props)  # type: ignore
    if (sum_ > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    return random_split(dataset, section_sizes, generator=generator)


class TrainingMode(Enum):
    epoch = auto()
    step = auto()


class BatchSamplerBase(Sampler[Sequence[int]]):
    def __init__(self, epoch_length: int | None = None) -> None:
        self.epoch_length = epoch_length

    @implements(Sampler)
    @abstractmethod
    def __iter__(self) -> Iterator[list[int]]:
        ...

    def __len__(self) -> float | int:
        """The number of samples drawn.
        If epoch_length is None then the sampler has no length defined and will
        be sampled from infinitely. However, in such cases, __len__ still needs to
        be defined for downstream compatibility (e.g. with PyTorch Lightning) and for
        this it suffices to simply return 'inf'.
        """
        if self.epoch_length is None:
            return math.inf
        return self.epoch_length


def _check_generator(generator: torch.Generator | None) -> torch.Generator:
    """ If the generator is None, randomly initialise a generator object."""
    if generator is None:
        generator = torch.Generator()
        generator = generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    return generator


def num_batches_per_epoch(num_samples: int, *, batch_size: int, drop_last: bool = False) -> int:
    rounding_fn = math.floor if drop_last else math.ceil
    return rounding_fn(num_samples / batch_size)


class SequentialBatchSampler(BatchSamplerBase):
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
        data_source (Sized): Object of the same size as the data to be sampled from.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        batch_size: int,
        training_mode: TrainingMode | str = TrainingMode.step,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self._dataset_size = len(data_source)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        if isinstance(training_mode, str):
            training_mode = str_to_enum(str_=training_mode, enum=TrainingMode)
        self.training_mode = training_mode
        if self.training_mode is TrainingMode.epoch:
            epoch_length = num_batches_per_epoch(
                num_samples=self._dataset_size, batch_size=self.batch_size, drop_last=self.drop_last
            )
        else:
            epoch_length = None
        super().__init__(epoch_length=epoch_length)

    def _generate_idx_seq(self, generator: torch.Generator) -> Tensor:
        """Generate a random sequence of unique indexes."""
        if self.shuffle:
            return torch.randperm(self._dataset_size, generator=generator)
        return torch.arange(self._dataset_size)

    def _batch_indexes(self, indexes: Tensor) -> Sequence[Tensor]:
        """Split the indexes into batches."""
        return indexes.split(self.batch_size)

    @implements(BatchSamplerBase)
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        batched_idxs_iter = iter(self._batch_indexes(self._generate_idx_seq(generator=generator)))
        # Iterate until some stopping criterion is reached
        while True:
            batch_idxs = next(batched_idxs_iter, None)
            if (batch_idxs is None) or (len(batch_idxs) < self.batch_size):
                if self.epoch_length is None:
                    new_idx_seq = self._generate_idx_seq(generator=generator)
                    if (batch_idxs is not None) and (not self.drop_last):
                        # Rather than dropping the last batch if it is incomplete or simply using it,
                        # incomplete as it may be, we take the alternative approach of concatenating the surplus
                        # batch to the beginning of the next generation of indexes
                        new_idx_seq = torch.cat([batch_idxs, new_idx_seq])
                    batched_idxs_iter = iter(self._batch_indexes(new_idx_seq))
                else:
                    if (batch_idxs is not None) and (not self.drop_last):
                        yield batch_idxs.tolist()
                    break
            else:
                yield batch_idxs.tolist()


class BaseSampler(Enum):
    random = auto()
    sequential = auto()


class StratifiedBatchSampler(BatchSamplerBase):
    r"""Samples equal proportion of elements from ``[0,..,len(group_ids)-1]``.

    To drop certain groups, set their multiplier to 0.

    Args:
        group_ids: a sequence of group IDs, not necessarily contiguous.
        num_samples_per_group: number of samples to draw per group. Note that if a multiplier is > 1
            then effectively more samples will be drawn for that group.
        replacement: if ``True``, samples are drawn with replacement. If not, they are drawn without
            replacement, which means that when a sample index is drawn for a row, it cannot be drawn
            again for that row.
        base_sampler: the base sampling strategy to use (sequential vs. random).
        shuffle. whether to shuffle the subsets of the data after each pass (only applicable when the
            base_sampler is set to ``sequential``).
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
        *,
        num_samples_per_group: int,
        multipliers: dict[int, int] | None = None,
        base_sampler: BaseSampler | str = BaseSampler.sequential,
        training_mode: TrainingMode | str = TrainingMode.step,
        replacement: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
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
        if isinstance(base_sampler, str):
            base_sampler = str_to_enum(str_=base_sampler, enum=BaseSampler)
        if isinstance(training_mode, str):
            training_mode = str_to_enum(str_=training_mode, enum=TrainingMode)

        self.num_samples_per_group = num_samples_per_group
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
        self.batch_size = self.num_groups_effective * self.num_samples_per_group
        self.sampler = base_sampler
        self.replacement = replacement
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.training_mode = training_mode

        if self.training_mode is TrainingMode.epoch:
            # We define the length of the sampler to be the maximum number of steps
            # needed to do a complete pass of a group's data
            groupwise_epoch_length = [
                num_batches_per_epoch(
                    num_samples=len(idxs),
                    batch_size=mult * num_samples_per_group,
                    drop_last=self.drop_last,
                )
                for idxs, mult in self.groupwise_idxs
            ]
            # Sort the groupwise-idxs by their associated epoch-length
            sorted_idxs_desc = np.argsort(groupwise_epoch_length)[::-1]
            self.groupwise_idxs = [self.groupwise_idxs[idx] for idx in sorted_idxs_desc]
            max_epoch_length = groupwise_epoch_length[sorted_idxs_desc[0]]
        else:
            max_epoch_length = None

        super().__init__(epoch_length=max_epoch_length)

    def _sequential_sampler(self, generator: torch.Generator) -> Iterator[list[int]]:
        samplers_and_idxs = [
            (
                iter(
                    SequentialBatchSampler(
                        data_source=group_idxs,
                        batch_size=self.num_samples_per_group * multiplier,
                        shuffle=self.shuffle,
                        generator=generator,
                        training_mode=self.training_mode
                        if self.training_mode is TrainingMode.epoch and (group_num == 0)
                        else TrainingMode.step,  # group-idxs are sorted by epoch-length
                    )
                ),
                group_idxs,
            )
            for group_num, (group_idxs, multiplier) in enumerate(self.groupwise_idxs)
            # Skip any groups with a non-positive multiplier
            if (multiplier > 0)
        ]
        sampled_idxs: list[int]
        # 'step' mode is enabled, making the sampling procedure very simple
        # because there's no need to special case the last batch
        if self.epoch_length is None:
            while True:
                sampled_idxs = []
                for sampler, group_idxs in samplers_and_idxs:
                    sampled_idxs.extend(group_idxs[next(sampler)])
                yield sampled_idxs
        # 'epoch' mode is enabled - handling the last batch is quite involved
        # as we need to preserve the ratio between the groups prescribed by the
        # multipliers
        else:
            # Factor by which to reduce the batch by - only relevant for the
            # last batch and is computed as the ratio of the number of
            # drawn for the final batch to the group-specific batch size
            # for the group with the longest epoch-length
            batch_reduction_factor: float | None = None
            for step in range(1, self.epoch_length + 1):
                sampled_idxs = []
                for group_num, (sampler, group_idxs) in enumerate(samplers_and_idxs):
                    idxs_of_idxs = next(sampler)
                    # The groups are ordered by epoch-length so we only need to check the first group
                    # (being the one that dictates the length of a epoch for the whole sampler)
                    if group_num == 0:
                        if step == self.epoch_length:
                            batch_reduction_factor = len(idxs_of_idxs) / (
                                self.num_samples_per_group * self.groupwise_idxs[group_num][1]
                            )
                            # The batch is incomplete and drop-last is enabled - terminte the iteration
                            if self.drop_last and (not batch_reduction_factor):
                                return
                    else:
                        if batch_reduction_factor is not None:
                            # Subsample the indexes according to the batch-reduction-factor
                            reduced_sample_count = round(len(idxs_of_idxs) * batch_reduction_factor)
                            idxs_of_idxs = idxs_of_idxs[:reduced_sample_count]
                    # Collate the indexes
                    sampled_idxs.extend(group_idxs[idxs_of_idxs])

                yield sampled_idxs

    def _random_sampler(self, generator: torch.Generator) -> Iterator[list[int]]:
        while True:
            sampled_idxs: list[Tensor] = []
            for group_idxs, multiplier in self.groupwise_idxs:
                if self.replacement:
                    for _ in range(multiplier):
                        # sampling with replacement:
                        # just sample enough random numbers to fill the quota
                        idxs_of_idxs = torch.randint(
                            low=0,
                            high=len(group_idxs),
                            size=(self.num_samples_per_group,),
                            generator=generator,
                        )
                        sampled_idxs.append(group_idxs[idxs_of_idxs])
                else:
                    # sampling without replacement:
                    # first shuffle the indexes and then take as many as we need
                    shuffled_idx = group_idxs[torch.randperm(len(group_idxs), generator=generator)]
                    # all elements in `sampled_idx` have to have the same size,
                    # so we split the tensor in equal-sized parts and then take as many as we need
                    chunks = torch.split(shuffled_idx, self.num_samples_per_group)
                    sampled_idxs += list(chunks[:multiplier])
            yield torch.cat(sampled_idxs, dim=0).tolist()

    @implements(BatchSamplerBase)
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        if self.sampler is BaseSampler.random:
            return self._random_sampler(generator=generator)
        else:
            return self._sequential_sampler(generator=generator)
