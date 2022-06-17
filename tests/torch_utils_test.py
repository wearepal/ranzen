import time

import numpy as np
import pytest
import torch
from torch import Tensor

from ranzen.torch.utils import Event, batchwise_pdist


def test_event() -> None:
    with Event() as event:
        time.sleep(0.10)
    pytest.approx(event.time, 0.10)
    print(event)


def test_batchwise_pdist() -> None:
    random = np.random.default_rng(888)
    x = torch.from_numpy(random.normal(size=(100, 9)))

    pdist_correct = torch.cdist(x, x)
    pdist_actual = batchwise_pdist(x, chunk_size=11)
    assert torch.allclose(pdist_correct, pdist_actual, rtol=1e-7, atol=1e-7)
    assert torch.allclose(_get_dists(x).sqrt(), pdist_actual, rtol=1e-7, atol=1e-7)


def _get_dists(embeddings: Tensor) -> Tensor:
    dist_mat = embeddings @ embeddings.t()
    sq = dist_mat.diagonal().view(embeddings.size(0), 1)
    return -2 * dist_mat + sq + sq.t()
