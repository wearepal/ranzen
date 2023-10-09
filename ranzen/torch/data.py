from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import (
    Any,
    Final,
    Generic,
    Iterator,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Sized,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)
from typing_extensions import Self, override

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Sampler

from ranzen.misc import some, str_to_enum

__all__ = [
    "ApproxStratBatchSampler",
    "BatchSamplerBase",
    "GreedyCoreSetSampler",
    "SequentialBatchSampler",
    "SizedDataset",
    "StratifiedBatchSampler",
    "Subset",
    "TrainTestSplit",
    "TrainingMode",
    "WeightedBatchSampler",
    "prop_random_split",
    "stratified_split_indices",
]


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class SizedDataset(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co:
        ...

    def __len__(self) -> int:
        ...


D = TypeVar("D", bound=SizedDataset)


class Subset(Generic[D]):
    r"""
    Subset of a dataset at specified indices.
    """

    dataset: D
    indices: Sequence[int]

    def __init__(self, dataset: D, indices: Sequence[int]) -> None:
        """
        :param dataset: The whole Dataset.
        :param indices: Indices in the whole set selected for subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


@overload
def prop_random_split(
    dataset_or_size: D,
    *,
    props: Sequence[float] | float,
    as_indices: Literal[False] = ...,
    seed: int | None = ...,
) -> list[Subset[D]]:
    ...


@overload
def prop_random_split(
    dataset_or_size: SizedDataset,
    *,
    props: Sequence[float] | float,
    as_indices: Literal[True],
    seed: int | None = ...,
) -> list[list[int]]:
    ...


@overload
def prop_random_split(
    dataset_or_size: int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = ...,
    seed: int | None = ...,
) -> list[list[int]]:
    ...


@overload
def prop_random_split(
    dataset_or_size: D | int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = ...,
    seed: int | None = ...,
) -> list[Subset[D]] | list[list[int]]:
    ...


def prop_random_split(
    dataset_or_size: D | int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = False,
    seed: int | None = None,
) -> list[Subset[D]] | list[list[int]]:
    """Splits a dataset based on proportions rather than on absolute sizes

    :param dataset_or_size: Dataset or size (length) of the dataset to split.
    :param props: The fractional size of each subset into which to randomly split the data.
        Elements must be non-negative and sum to 1 or less; if less then the size of the final
        split will be computed by complement.

    :param as_indices: If ``True`` the raw indices are returned instead of subsets constructed
        from them when `dataset_or_len` is a dataset. This means that when `dataset_or_len`
        corresponds to the length of a dataset, this argument has no effect and
        the function always returns the split indices.

    :param seed: The PRNG used for determining the random splits.

    :returns: Random subsets of the data of the requested proportions.

    :raises ValueError: If the dataset does not have a ``__len__`` method or sum(props) > 1.
    """
    if isinstance(dataset_or_size, int):
        len_ = dataset_or_size
    else:
        if not hasattr(dataset_or_size, "__len__"):
            raise ValueError(
                "Split proportions can only be computed for datasets with __len__ defined."
            )
        len_ = len(dataset_or_size)

    if isinstance(props, (float, int)):
        props = [props]
    sum_ = np.sum(props)
    if (sum_ > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if sum_ < 1:
        section_sizes.append(len_ - sum(section_sizes))
    generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    indices = torch.randperm(sum(section_sizes), generator=generator).tolist()
    splits = [
        indices[offset - length : offset]
        for offset, length in zip(np.cumsum(section_sizes), section_sizes)
    ]

    if as_indices or isinstance(dataset_or_size, int):
        return splits
    return [Subset(dataset_or_size, indices=split) for split in splits]


_S = TypeVar("_S")


@dataclass(frozen=True)
class TrainTestSplit(Generic[_S]):
    train: _S
    test: _S

    def __iter__(self) -> Iterator[_S]:
        yield from (self.train, self.test)


def stratified_split_indices(
    labels: Tensor | npt.NDArray[np.int_] | Sequence[int],
    *,
    default_train_prop: float,
    train_props: dict[int, float] | None = None,
    seed: int | None = None,
) -> TrainTestSplit[list[int]]:
    """Splits the data into train/test sets conditional on super- and sub-class labels.

    :param labels: Tensor, array or sequence encoding the label associated with each sample.
    :param default_train_prop: Proportion of samples for a given to sample for
        the training set for those y-s combinations not specified in ``train_props``.

    :param train_props: Proportion of each group  to sample for the training set.
        If ``None`` then the function reduces to a simple random split of the data.

    :param seed: PRNG seed to use for sampling.

    :returns: Train-test split.

    :raises ValueError: If a value in ``train_props`` is not in the range [0, 1] or if a key is not
        present in ``group_ids``.
    """
    if not isinstance(labels, Tensor):
        labels = torch.as_tensor(labels, dtype=torch.long)
    # Initialise the random-number generator
    generator = torch.default_generator if seed is None else torch.Generator().manual_seed(seed)
    groups, label_counts = labels.unique(return_counts=True)
    train_props_all = dict.fromkeys(groups.tolist(), default_train_prop)

    if train_props is not None:
        for label, train_prop in train_props.items():
            if not 0 <= train_prop <= 1:
                raise ValueError(
                    "All splitting proportions specified in 'train_props' must be in the "
                    "range [0, 1]."
                )
            if label not in groups:
                raise ValueError(f"No samples belonging to the group in 'group_ids'.")
            train_props_all[label] = train_prop

    # Shuffle the samples before sampling
    perm_inds = torch.randperm(len(labels), generator=generator)
    labels_perm = labels[perm_inds]

    sort_inds = labels_perm.sort(dim=0, stable=True).indices
    thresholds = cast(
        Tensor, (torch.as_tensor(tuple(train_props_all.values())) * label_counts).round().long()
    )
    thresholds = torch.stack([thresholds, label_counts], dim=-1)
    thresholds[1:] += label_counts.cumsum(0)[:-1].unsqueeze(-1)

    train_test_inds = sort_inds.tensor_split(thresholds.flatten()[:-1], dim=0)
    train_inds = perm_inds[torch.cat(train_test_inds[0::2])].tolist()
    test_inds = perm_inds[torch.cat(train_test_inds[1::2])].tolist()

    return TrainTestSplit(train=train_inds, test=test_inds)


class TrainingMode(Enum):
    """An enum for the training mode."""

    epoch = auto()
    """epoch-based training"""
    step = auto()
    """step-based training"""


class BatchSamplerBase(Sampler[List[int]]):
    def __init__(self, epoch_length: int | None = None) -> None:
        self.epoch_length: Final[int | None] = epoch_length

    @override
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
    """If the generator is None, randomly initialise a generator object."""
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

    :param data_source: Object of the same size as the data to be sampled from.
    :param batch size: How many samples per batch to load.
    :param shuffle: Set to ``True`` to have the data reshuffled at every epoch.
    :param drop_last: Set to ``True`` to drop the last incomplete batch,
    :param shuffle: Set to ``True`` to have the data reshuffled
    :param generator: Pseudo-random-number generator to use for shuffling the dataset.

    :example:
        >>> batch_sampler = InfSequentialBatchSampler(data_source=train_data, batch_size=100, shuffle=True)
        >>> train_loader = DataLoader(train_data, batch_sampler=batch_sampler, shuffle=False, drop_last=False) # drop_last and shuffle need to be False
        >>> train_loader_iter = iter(train_loader)
        >>> for _ in range(train_iters):
        >>>     batch = next(train_loader_iter)
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
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        if isinstance(training_mode, str):
            training_mode = str_to_enum(str_=training_mode, enum=TrainingMode)
        self.training_mode = training_mode
        if self.training_mode is TrainingMode.epoch:
            epoch_length = num_batches_per_epoch(
                num_samples=len(self.data_source),
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            )
        else:
            epoch_length = None
        super().__init__(epoch_length=epoch_length)

    def _generate_idx_seq(self, generator: torch.Generator) -> Tensor:
        """Generate a random sequence of unique indexes."""
        if self.shuffle:
            return torch.randperm(len(self.data_source), generator=generator)
        return torch.arange(len(self.data_source))

    def _batch_indexes(self, indexes: Tensor) -> Sequence[Tensor]:
        """Split the indexes into batches."""
        return indexes.split(self.batch_size)

    @override
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
    """An enum for the base-sampler to use for StratifiedBatchSampler."""

    random = auto()
    """random sampler"""
    sequential = auto()
    """sequential sampler"""


class StratifiedBatchSampler(BatchSamplerBase):
    r"""Samples equal proportion of elements from ``[0,..,len(group_ids)-1]``.

    To drop certain groups, set their multiplier to 0.

    :param group_ids: A sequence of group IDs, not necessarily contiguous.

    :param num_samples_per_group: Number of samples to draw per group. Note that if a multiplier is > 1
        then effectively more samples will be drawn for that group.

    :param multipliers: An optional dictionary that maps group IDs to multipliers. If a multiplier is
        greater than 1, the corresponding group will be sampled at twice the rate as the other
        groups. If a multiplier is 0, the group will be skipped.

    :param base_sampler: The base sampling strategy to use (sequential vs. random).

    :param replacement: if ``True``, samples are drawn with replacement. If not, they are drawn without
        replacement, which means that when a sample index is drawn for a row, it cannot be drawn
        again for that row.


    :param shuffle: Whether to shuffle the subsets of the data after each pass (only applicable when the
        base_sampler is set to ``sequential``).

    :param drop_last: Set to ``True`` to drop the last (on a per-group basis) incomplete batch.
    :param generator: Pseudo-random-number generator to use for shuffling the dataset.

    :example:
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
                    sampled_idxs.extend(group_idxs[next(sampler)].tolist())
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
                            # The batch is incomplete and drop-last is enabled - terminate the iteration
                            if self.drop_last and (not batch_reduction_factor):
                                return
                    else:
                        if batch_reduction_factor is not None:
                            # Subsample the indexes according to the batch-reduction-factor
                            reduced_sample_count = round(len(idxs_of_idxs) * batch_reduction_factor)
                            idxs_of_idxs = idxs_of_idxs[:reduced_sample_count]
                    # Collate the indexes
                    sampled_idxs.extend(group_idxs[idxs_of_idxs].tolist())

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

    @override
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        if self.sampler is BaseSampler.random:
            return self._random_sampler(generator=generator)
        else:
            return self._sequential_sampler(generator=generator)


class GreedyCoreSetSampler(BatchSamplerBase):
    r"""Constructs batches from 'oversampled' batches through greedy core-set approximation.

    Said approximation takes the form of the furtherst-frist traversal (FFT) algorithm.

    :param batch_size: Budget for the core-set,

    :param embeddings: Embedded dataset from which to sample the core-sets according;
        the order of the embeddings, v, must match the order of the dataset
        (i.e. f(x_i) = v_i, for embedding function f and inputs x)

    :param oversampling_factor: How many times larger than the budget the batch to be sampled from

        should be.
    :param generator: Pseudo-random-number generator to use for shuffling the dataset.
    """

    def __init__(
        self,
        embeddings: Tensor,
        *,
        batch_size: int,
        oversampling_factor: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.oversampling_factor = oversampling_factor
        self.embeddings = embeddings.flatten(start_dim=1).detach().cpu()
        self.budget = batch_size
        self._num_oversampled_samples = min(
            self.budget * self.oversampling_factor, len(self.embeddings)
        )
        self.generator = generator

        super().__init__(epoch_length=None)

    def _get_dists(self, batch_idxs: Tensor) -> Tensor:
        batch = self.embeddings[batch_idxs]
        dist_mat = batch @ batch.t()
        sq = dist_mat.diagonal().view(batch.size(0), 1)
        return -2 * dist_mat + sq + sq.t()

    @override
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        # iterative forever (until some externally defined stopping-criterion is reached)
        while True:
            # First sample the 'oversampled' batch from which to construct the core-set
            os_batch_idxs = torch.randperm(len(self.embeddings), generator=generator)[
                : self._num_oversampled_samples
            ]
            # Compute the euclidean distance between all pairs in said batch
            dists = self._get_dists(os_batch_idxs)
            # Mask indicating whether a sample is still yet to be sampled (1=unsampled, 0=sampled)
            # - updating a mask is far more efficnet than reconstructing the list of unsampled
            # indexes every iteration (however, we do have to be careful about the 'meta-indexing'
            # it introduces)
            unsampled_m = torch.ones_like(os_batch_idxs, dtype=torch.bool)
            # there's no obvious decision rule for seeding the core-set so we just select the first
            # point arbitrarily, moving it from the unsampled pool to the sampled one by setting
            # its corresponding mask-value to 0
            sampled_idxs = [int(os_batch_idxs[0])]
            unsampled_m[0] = 0

            # Begin the furthest-first traversal algorithm
            while len(sampled_idxs) < self.budget:
                # p := argmax min_{i\inB}(d(x, x_i)); i.e. select the point which maximizes the
                # minimum squared Euclidean-distance to all previously selected points
                # NOTE: The argmax index is relative to the unsampled indexes
                rel_idx = torch.argmax(torch.min(dists[~unsampled_m][:, unsampled_m], dim=0).values)
                # Retiieve the index corresponding to the previously-computed argmax index
                to_sample = os_batch_idxs[unsampled_m][rel_idx]
                sampled_idxs.append(int(to_sample))
                # Update the mask, which corresponds to moving the sampled index from the unsampled
                # pool to the sampled pool
                unsampled_m[unsampled_m.nonzero()[rel_idx]] = 0

            yield sampled_idxs


class WeightedBatchSampler(BatchSamplerBase):
    r"""Implements a batch-sampler version of :class:`torch.utils.data.WeightedRandomSampler`."""

    def __init__(
        self,
        weights: Sequence[float] | Tensor,
        *,
        batch_size: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        :param batch_size: Number of samples to draw per batch/iteration.
        :param weights:  A sequence or tensor of weights, not necessarily summing to one.
        :param replacement: If ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a sample index is drawn
            for a row, it cannot be drawn again for that row.

        :param generator: Pseudo-random-number generator to use for randomly sampling indexes.

        :raises ValueError: If ``batch_size`` is non-positive or is greater than the number of weights
            when ``replacement=False``
        """
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer, but got batch_size={batch_size}."
            )
        if (not replacement) and ((len_ := len(weights)) < batch_size):
            raise ValueError(
                "batch_size cannot be greater than the number of weights if sampling "
                f"without replacement, but got batch_size={batch_size} and weights of length {len_}"
            )
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.batch_size = batch_size
        self.replacement = replacement
        self.generator = generator

        super().__init__(epoch_length=None)

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[int] | Tensor,
        *,
        batch_size: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> Self:
        """
        Instantiate a :class:`~WeightedBatchSampler` from a sequenece or tensor of ints, where
        ``weights`` is computed using the inverse frequencies of the values in ``labels``.

        :param labels: Labels from which to compute the sample weights from; should be of length
            equal to the size of the associated dataset being indexed.

        :param batch_size: Number of samples to draw per batch/iteration.
        :param replacement: If ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

        :param generator: Pseudo-random-number generator to use for randomly sampling indexes.

        :returns: A :class:`~WeightedBatchSampler` instance with ``weights`` computed using the
            inverse frequencies of the values in ``labels``.

        :raises ValueError: If ``labels`` is a tensor and does not have dtype :class:`torch.long`.
        """
        if isinstance(labels, Tensor):
            if labels.dtype is not torch.long:
                raise ValueError("labels must have dtype 'long' ('int64').")
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
        _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
        counts_r = counts.reciprocal()
        sample_weights = counts_r / counts_r.sum()
        weights = sample_weights[inverse].squeeze()
        return cls(
            weights=weights,
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )

    @override
    def __iter__(self) -> Iterator[list[int]]:
        generator = _check_generator(self.generator)
        # Iterate until some stopping criterion is reached
        while True:
            yield torch.multinomial(
                self.weights,
                num_samples=self.batch_size,
                replacement=self.replacement,
                generator=generator,
            ).tolist()


S = NewType("S", int)  # type meant to help with documentation
Y = NewType("Y", int)


class ApproxStratBatchSampler(BatchSamplerBase):
    r"""Approximate Stratified Batch Sampler.

    Essentially, we’re doing: :math:`x\sim P(x|s,y)` where :math:`s\sim \text{uniform}(S|y)`. That
    is, we iterate over all classes y and uniformly sample a subgroup s, and then we sample a
    datapoint from that s-y combination.

    You have to either specify ``num_samples_per_group`` or ``num_samples_per_class`` (but not
    both).

    If ``num_samples_per_group`` is given, this faithfully implements the :math:`\pi` function. This means
    that for those classes which have "full s-support" (all subgroups are present), we don’t sample
    a subgroup but iterate over each subgroup one-by-one. We take ``num_samples_per_group`` samples
    from each s-y combination.

    On the other hand, if ``num_samples_per_class`` is given, then classes with full s-support are
    not given special treatment. We always sample as many subgroups as are specified in
    ``num_samples_per_class`` and then take a single datapoint from each of these s-y combinations.

    :param class_labels: List-like object with the class labels.
    :param subgroup_labels: List-like object with the subgroup labels.
    :param num_samples_per_group: How many samples to take per s-y group. Cannot be specified
        together with ``num_samples_per_class``.
    :param num_samples_per_class: How many samples to take per y class. Cannot be specified
        together with ``num_samples_per_group``.
    :param training_mode: Iteration-based vs epoch-based.
    :param generator: Torch generator for random numbers.
    """

    def __init__(
        self,
        class_labels: Sequence[int],
        subgroup_labels: Sequence[int],
        *,
        num_samples_per_group: Optional[int] = None,
        num_samples_per_class: Optional[int] = None,
        training_mode: TrainingMode = TrainingMode.step,
        generator: Union[torch.Generator, None] = None,
    ) -> None:
        if some(num_samples_per_group) and some(num_samples_per_class):
            raise ValueError(
                "Specify either `num_samples_per_group` or `num_samples_per_class` but not both"
            )
        elif (num_samples_per_group is None) and (num_samples_per_class is None):
            raise ValueError("Specify one of `num_samples_per_group` or `num_samples_per_class`")

        assert len(class_labels) == len(subgroup_labels), "labels should have the same length"

        class_labels_t = torch.as_tensor(class_labels, dtype=torch.int64)
        subgroup_labels_t = torch.as_tensor(subgroup_labels, dtype=torch.int64)
        # find all unique labels
        classes = class_labels_t.unique().tolist()
        subgroups = subgroup_labels_t.unique().tolist()
        # cast to nice-looking types
        classes = cast(List[Y], classes)
        subgroups = cast(List[S], subgroups)

        # get the indexes for each group separately and store them in a hierarchical dict
        groupwise_idxs: defaultdict[Y, list[Tensor]] = defaultdict(list)
        for class_ in classes:
            for subgroup in subgroups:
                # Idxs needs to be 1 dimensional
                idxs = (
                    ((subgroup_labels_t == subgroup) & (class_labels_t == class_))
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
                num_samples = len(idxs)

                if num_samples == 0:
                    continue
                groupwise_idxs[class_].append(idxs)

                if some(num_samples_per_group) and (num_samples < num_samples_per_group):
                    raise ValueError(
                        f"Not enough samples in group (s={subgroup}, y={class_}) "
                        f"to sample {num_samples_per_group} (available: {num_samples})."
                    )

        self.groupwise_idxs = groupwise_idxs
        self.generator = generator
        if some(num_samples_per_group):
            # In each batch, we want the full list of subgroups
            self.num_subgroup_samples = len(subgroups)
            self.num_samples_per_group = num_samples_per_group
            self.classes_with_full_support: set[Y] = {
                y
                for y, subgroup_idxs in groupwise_idxs.items()
                if len(subgroup_idxs) == self.num_subgroup_samples
            }
        elif some(num_samples_per_class):
            self.num_subgroup_samples = num_samples_per_class
            self.num_samples_per_group = 1  # Take one sample for each sampled s
            self.classes_with_full_support = set()  # No special-casing for full-support classes
        else:
            raise RuntimeError("shouldn’t happen")

        self.batch_size = len(classes) * self.num_subgroup_samples * self.num_samples_per_group

        if training_mode is TrainingMode.epoch:
            # some groups have fewer samples than others
            # we want to know how many batches to sample to cover even the biggest group
            if some(num_samples_per_group):
                max_epoch_length = max(
                    num_batches_per_epoch(
                        num_samples=len(idxs), batch_size=num_samples_per_group, drop_last=False
                    )
                    for subgroup_idxs in self.groupwise_idxs.values()
                    for idxs in subgroup_idxs
                )
            elif some(num_samples_per_class):
                # (the following calculation is a bit sketchy, but I think it's fine)
                max_epoch_length = max(
                    num_batches_per_epoch(
                        num_samples=max(len(idxs) for idxs in subgroup_idxs) * len(subgroup_idxs),
                        batch_size=num_samples_per_class,
                        drop_last=False,
                    )
                    for subgroup_idxs in self.groupwise_idxs.values()
                )
            else:
                raise RuntimeError("shouldn’t happen")
        else:
            max_epoch_length = None

        super().__init__(epoch_length=max_epoch_length)

    @override
    def __iter__(self) -> Iterator[list[int]]:
        generator = self.generator
        if generator is None:
            generator = torch.Generator()
            generator = generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        while True:
            sampled_idxs: list[Tensor] = []
            for y, subgroupwise_idxs in self.groupwise_idxs.items():
                if y in self.classes_with_full_support:
                    # just take samples from each subgroup
                    sampled_idxs.extend(
                        self._take_samples_per_group(idxs, generator) for idxs in subgroupwise_idxs
                    )
                else:
                    # first sample (with replacement) the required number of subgroups
                    subgroups = torch.randint(
                        low=0,
                        high=len(subgroupwise_idxs),
                        size=(self.num_subgroup_samples,),
                        generator=generator,
                    )
                    # then take samples from each
                    sampled_idxs.extend(
                        self._take_samples_per_group(subgroupwise_idxs[i], generator)
                        for i in subgroups.tolist()
                    )
            yield torch.cat(sampled_idxs, dim=0).tolist()

    def _take_samples_per_group(self, tensor: Tensor, generator: torch.Generator) -> Tensor:
        # first shuffle and then take as many as we need
        shuffled = tensor[torch.randperm(len(tensor), generator=generator)]
        return shuffled[: self.num_samples_per_group]
