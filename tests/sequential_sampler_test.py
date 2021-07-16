import pytest
import torch
from torch import Tensor

from kit.torch import SequentialBatchSampler


@pytest.fixture(scope="module")
def data() -> Tensor:  # type: ignore[no-any-unimported]
    return torch.arange(200)


def test_regeneration(data: Tensor) -> None:
    batch_size = 175
    dataset_size = 200
    sampler = SequentialBatchSampler(data_source=data, batch_size=batch_size, shuffle=False)
    sampler_iter = iter(sampler)
    indexes = torch.as_tensor(next(sampler_iter))
    assert len(indexes) == batch_size
    assert (indexes == data[:batch_size]).all()

    indexes = torch.as_tensor(next(sampler_iter))
    assert len(indexes) == batch_size
    residual = dataset_size - batch_size
    assert (indexes[:residual] == data[-residual:]).all()


@pytest.mark.parametrize("drop_last", [True, False])
def test_sized(data: Tensor, drop_last: bool) -> None:
    sampler = SequentialBatchSampler(
        data_source=data, batch_size=55, shuffle=False, sized=True, drop_last=drop_last
    )
    batches = [batch for batch in sampler]
    assert len(batches) == sampler.epoch_length
    assert len(batches) == (4 - drop_last)
