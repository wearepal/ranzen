from __future__ import annotations
from itertools import islice

import pytest
import torch
from torch.utils.data import TensorDataset

from ranzen.torch import prop_random_split
from ranzen.torch.data import ApproxStratBatchSampler, Subset, stratified_split_indices


@pytest.fixture(scope="module")
def dummy_ds() -> TensorDataset:
    return TensorDataset(torch.randn(100))


@pytest.mark.parametrize("as_indices", [False, True])
@pytest.mark.parametrize("props", [0.5, [-0.2, 0.5], [0.1, 0.3, 0.4], [0.5, 0.6]])
def test_prop_random_split(
    dummy_ds: TensorDataset, props: float | list[float], as_indices: bool
) -> None:
    sum_ = props if isinstance(props, (float, int)) else sum(props)
    props_ls = [props] if isinstance(props, (float, int)) else props
    if sum_ > 1 or any(not (0 <= prop <= 1) for prop in props_ls):
        with pytest.raises(ValueError):
            splits = prop_random_split(dataset_or_size=dummy_ds, props=props, as_indices=as_indices)
    else:
        splits = prop_random_split(dataset_or_size=dummy_ds, props=props, as_indices=as_indices)
        sizes = [len(split) for split in splits]
        sum_sizes = sum(sizes)
        assert len(splits) == (len(props_ls) + 1)
        assert sum_sizes == len(dummy_ds)
        assert sizes[-1] == (len(dummy_ds) - (round(sum_ * len(dummy_ds))))
        if not as_indices:
            assert all(isinstance(split, Subset) for split in splits)

        splits = prop_random_split(
            dataset_or_size=len(dummy_ds), props=props, as_indices=as_indices
        )
        sizes = [len(split) for split in splits]
        sum_sizes = sum(sizes)
        assert len(splits) == (len(props_ls) + 1)
        assert sum_sizes == len(dummy_ds)
        assert sizes[-1] == (len(dummy_ds) - (round(sum_ * len(dummy_ds))))


def test_stratified_split_indices() -> None:
    labels = torch.randint(low=0, high=4, size=(50,))
    train_inds, test_inds = stratified_split_indices(labels=labels, default_train_prop=0.5)
    labels_tr = labels[train_inds]
    labels_te = labels[test_inds]
    counts_tr = labels_tr.unique(return_counts=True)[1]
    counts_te = labels_te.unique(return_counts=True)[1]
    assert torch.isclose(counts_tr, counts_te, atol=1).all()

    train_props = {0: 0.25, 1: 0.45}
    train_inds, test_inds = stratified_split_indices(
        labels=labels,
        default_train_prop=0.5,
        train_props=train_props,
    )
    labels_tr = labels[train_inds]
    labels_te = labels[test_inds]

    for label, train_prop in train_props.items():
        train_m = labels_tr == label
        test_m = labels_te == label
        all_m = labels == label

        n_train = train_m.count_nonzero().item()
        n_test = test_m.count_nonzero().item()
        n_all = all_m.count_nonzero().item()

        assert n_train == pytest.approx(train_prop * n_all, abs=1)
        assert n_test == pytest.approx((1 - train_prop) * n_all, abs=1)


def test_approximate_stratified_sampler() -> None:
    class_labels = [0, 1, 1, 0, 1]
    subgroup_labels = [1, 1, 1, 0, 1]
    generator = torch.Generator()
    generator = generator.manual_seed(42)

    sampler = ApproxStratBatchSampler(
        class_labels, subgroup_labels, num_samples_per_group=1, generator=generator
    )
    assert sampler.batch_size == 4
    assert len(sampler.classes_with_full_support) == 1

    batches = list(islice(sampler, 100))
    # all batches should have 4 elements
    assert all(len(batch) == 4 for batch in batches)
    # element 0 or element 3 has to be in all batches, because they're the only ones with y=0
    assert all((0 in batch or 3 in batch) for batch in batches)
    # every batch has two of elements 1, 2, 4, because they're the only ones with y=1
    assert all((batch[2] in {1, 2, 4} and batch[3] in {1, 2, 4}) for batch in batches)
    # all elements appear at least once
    # (this is not guaranteed, but with 100 samples, it is overwhelmingly likely; 1:2^100)
    assert all(any((i in batch) for batch in batches) for i in range(5))


def test_approximate_stratified_sampler_class() -> None:
    class_labels = [0, 1, 1, 0, 1]
    subgroup_labels = [1, 1, 1, 0, 1]
    generator = torch.Generator()
    generator = generator.manual_seed(42)

    sampler = ApproxStratBatchSampler(
        class_labels, subgroup_labels, num_samples_per_class=1, generator=generator
    )
    assert sampler.batch_size == 2
    assert len(sampler.classes_with_full_support) == 0

    batches = list(islice(sampler, 100))
    # all batches should have 2 elements
    assert all(len(batch) == 2 for batch in batches)
    # element 0 or element 3 has to be in all batches, because they're the only ones with y=0
    assert all((0 in batch or 3 in batch) for batch in batches)
    # every batch has one of elements 1, 2, 4, because they're the only ones with y=1
    assert all((batch[1] in {1, 2, 4}) for batch in batches)
    # all elements appear at least once
    # (this is not guaranteed, but with 100 samples, it is overwhelmingly likely; 1:2^100)
    assert all(any((i in batch) for batch in batches) for i in range(5))
