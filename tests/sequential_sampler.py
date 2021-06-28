import torch

from kit.torch import InfSequentialBatchSampler


def test_regeneration() -> None:
    batch_size = 175
    dataset_size = 200
    data = torch.arange(dataset_size)
    sampler = InfSequentialBatchSampler(data_source=data, batch_size=batch_size, shuffle=False)
    sampler_iter = iter(sampler)
    indexes = torch.as_tensor(next(sampler_iter))
    assert len(indexes) == batch_size
    assert (indexes == data[:batch_size]).all()

    indexes = torch.as_tensor(next(sampler_iter))
    assert len(indexes) == batch_size
    residual = dataset_size - batch_size
    assert (indexes[:residual] == data[-residual:]).all()
