from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from kit.torch import prop_random_split


@pytest.fixture(scope="module")
def dummy_ds() -> TensorDataset:
    return TensorDataset(torch.randn(100))


@pytest.mark.parametrize("props", [0.5, [-0.2, 0.5], [0.1, 0.3, 0.4], [0.5, 0.6]])
def test_prop_random_split(dummy_ds: TensorDataset, props: float | list[float]):
    sum_ = props if isinstance(props, float) else sum(props)
    props_ls = [props] if isinstance(props, float) else props
    if sum_ > 1 or any((not (0 <= prop <= 1)) for prop in props_ls):
        with pytest.raises(ValueError):
            splits = prop_random_split(dataset=dummy_ds, props=props)
    else:
        splits = prop_random_split(dataset=dummy_ds, props=props)
        sizes = [len(split) for split in splits]
        sum_sizes = sum(sizes)
        assert len(splits) == (len(props_ls) + 1)
        assert sum_sizes == len(dummy_ds)
        assert sizes[-1] == (len(dummy_ds) - (round(sum_ * len(dummy_ds))))
