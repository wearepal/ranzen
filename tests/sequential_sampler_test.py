import pytest
import torch
from torch import Tensor

from ranzen.torch import SequentialBatchSampler
from ranzen.torch.data import TrainingMode


@pytest.fixture(scope="module")
def data() -> Tensor:  # type: ignore[no-any-unimported]
    return torch.arange(200)


def test_regeneration(data: Tensor) -> None:  # type: ignore[no-any-unimported]
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
@pytest.mark.parametrize("batch_size", [55, 50])
def test_epoch_mode(data: Tensor, batch_size: int, drop_last: bool) -> None:  # type: ignore[no-any-unimported]
    sampler = SequentialBatchSampler(
        data_source=data,
        batch_size=batch_size,
        shuffle=False,
        training_mode=TrainingMode.epoch,
        drop_last=drop_last,
    )
    batches = [batch for batch in sampler]
    assert len(batches) == len(sampler)  # type: ignore
