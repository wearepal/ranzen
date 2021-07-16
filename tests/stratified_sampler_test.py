from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kit.torch.data import BaseSampler, StratifiedSampler


def count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]


@pytest.fixture
def group_ids() -> list[int]:
    return torch.cat(
        [torch.full((100,), 0), torch.full((200,), 1), torch.full((400,), 2), torch.full((800,), 3)]
    ).tolist()


@pytest.mark.parametrize("sampler", ["sequential", "random"])
def test_simple(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedSampler(
                group_ids,
                num_samples_per_group=num_samples_per_group,
                replacement=True,
                multipliers=None,
                base_sampler=sampler,
            )
        )
    )
    group_ids_t = torch.as_tensor(group_ids)
    _, counts = group_ids_t[indexes].unique(return_counts=True)
    assert len(indexes) == 4 * num_samples_per_group
    assert all(count == num_samples_per_group for count in counts)


def test_without_replacement(group_ids: list[int]) -> None:
    num_samples_per_group = 100
    indexes = next(
        iter(
            StratifiedSampler(
                group_ids,
                num_samples_per_group=num_samples_per_group,
                replacement=True,
                multipliers=None,
            )
        )
    )
    group_ids_t = torch.as_tensor(group_ids)
    _, counts = group_ids_t[indexes].unique(return_counts=True)
    assert len(indexes) == 4 * num_samples_per_group
    assert all(count == num_samples_per_group for count in counts)


@pytest.mark.parametrize("sampler", ["sequential", "random"])
def test_with_multipliers(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedSampler(
                group_ids,
                num_samples_per_group=num_samples_per_group,
                replacement=True,
                multipliers={0: 2, 1: 0, 2: 3},
                base_sampler=sampler,
            )
        ),
    )
    group_ids_t = torch.as_tensor(group_ids)
    assert len(indexes) == (2 + 0 + 3 + 1) * num_samples_per_group
    samples = group_ids_t[indexes]
    assert (samples == 0).sum() == (2 * num_samples_per_group)
    assert (samples == 1).sum() == 0
    assert (samples == 2).sum() == (3 * num_samples_per_group)


@pytest.mark.parametrize("sampler", ["sequential", "random"])
def test_with_dataloader(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 100
    batch_size = num_samples_per_group * 4
    batch_sampler = StratifiedSampler(
        group_ids,
        num_samples_per_group=num_samples_per_group,
        replacement=False,
        multipliers=None,
        base_sampler=sampler,
    )
    ds = TensorDataset(torch.as_tensor(group_ids))
    dl = DataLoader(dataset=ds, batch_sampler=batch_sampler, drop_last=False, shuffle=False)  # type: ignore
    iters = 0
    for (x,) in dl:
        assert x.size(0) == batch_size
        # assert all groups appear in the same quantity
        for i in range(0, 4):
            assert (x == i).sum() == num_samples_per_group
        iters += 1
        if iters == 2:
            break
