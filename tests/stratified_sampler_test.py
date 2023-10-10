import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ranzen.torch.data import BaseSampler, StratifiedBatchSampler, TrainingMode


@pytest.fixture
def group_ids() -> list[int]:
    return torch.cat(
        [torch.full((100,), 0), torch.full((200,), 1), torch.full((400,), 2), torch.full((800,), 3)]
    ).tolist()


@pytest.mark.parametrize("sampler", [list(BaseSampler)])
def test_simple(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedBatchSampler(
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
            StratifiedBatchSampler(
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


@pytest.mark.parametrize("sampler", [list(BaseSampler)])
def test_with_multipliers(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 800
    indexes = next(
        iter(
            StratifiedBatchSampler(
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


@pytest.mark.parametrize("sampler", [list(BaseSampler)])
def test_with_dataloader(group_ids: list[int], sampler: BaseSampler) -> None:
    num_samples_per_group = 100
    batch_size = num_samples_per_group * 4
    batch_sampler = StratifiedBatchSampler(
        group_ids,
        num_samples_per_group=num_samples_per_group,
        replacement=False,
        multipliers=None,
        base_sampler=sampler,
    )
    dataset = TensorDataset(torch.as_tensor(group_ids))
    data_loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, drop_last=False, shuffle=False
    )
    iters = 0
    for (x,) in data_loader:
        assert x.size(0) == batch_size
        # assert all groups appear in the same quantity
        for i in range(0, 4):
            assert (x == i).sum() == num_samples_per_group
        iters += 1
        if iters == 2:
            break


@pytest.mark.parametrize("drop_last", [True, False])
def test_sized(group_ids: list[int], drop_last: bool) -> None:
    sampler = StratifiedBatchSampler(
        group_ids=group_ids,
        num_samples_per_group=225,
        multipliers=None,
        shuffle=False,
        training_mode=TrainingMode.epoch,
        drop_last=drop_last,
    )
    batches = list(sampler)
    assert len(batches) == len(sampler)  # type: ignore
    assert len(batches) == (4 - drop_last)
