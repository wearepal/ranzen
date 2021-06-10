from __future__ import annotations

import numpy as np
import pytest
import torch
from typing import List
from torch.utils.data import DataLoader, TensorDataset

from kit.torch.data import StratifiedSampler


def count_true(mask: np.ndarray) -> int:
    """Count the number of elements that are True."""
    return mask.nonzero()[0].shape[0]


@pytest.fixture
def group_ids() -> List[int]:
    return torch.cat(
        [torch.full((100,), 0), torch.full((200,), 1), torch.full((400,), 2), torch.full((800,), 3)]
    ).tolist()


def test_simple(group_ids: List[int]) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedSampler(group_ids, num_samples_per_group, replacement=True, multipliers=None)
        )
    )
    indexes = np.array(indexes)
    assert len(indexes) == 4 * num_samples_per_group
    assert count_true(indexes < 100) == num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == num_samples_per_group
    assert count_true((300 <= indexes) & (indexes < 700)) == num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group


def test_without_replacement(group_ids: List[int]) -> None:
    num_samples_per_group = 100
    indexes = next(
        iter(
            StratifiedSampler(group_ids, num_samples_per_group, replacement=True, multipliers=None)
        )
    )
    indexes = np.array(indexes)

    assert len(indexes) == 4 * num_samples_per_group
    assert count_true(indexes < 100) == num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == num_samples_per_group
    assert count_true((300 <= indexes) & (indexes < 700)) == num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group


def test_with_multipliers(group_ids: List[int]) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedSampler(
                group_ids, num_samples_per_group, replacement=True, multipliers={0: 2, 1: 0, 2: 3}
            )
        ),
    )
    indexes = np.array(indexes)

    assert len(indexes) == (2 + 0 + 3 + 1) * num_samples_per_group
    assert count_true(indexes < 100) == 2 * num_samples_per_group
    assert count_true((100 <= indexes) & (indexes < 300)) == 0
    assert count_true((300 <= indexes) & (indexes < 700)) == 3 * num_samples_per_group
    assert count_true((700 <= indexes) & (indexes < 1500)) == num_samples_per_group


def test_with_dataloader(group_ids: List[int]) -> None:
    num_samples_per_group = 100
    batch_size = num_samples_per_group * 4
    batch_sampler = StratifiedSampler(
        group_ids, num_samples_per_group=num_samples_per_group, replacement=False, multipliers=None
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
