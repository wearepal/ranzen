from dataclasses import dataclass
from enum import auto

import pytest

from ranzen import AddDict, StrEnum, flatten_dict, gcopy, reproducible_random_split


def test_flatten_dict() -> None:
    d = {"a": 3, "b": {"c": 7.0, "d": {"e": True}, "f": -3}, "g": "hi"}
    assert flatten_dict(d) == {"a": 3, "b.c": 7.0, "b.d.e": True, "b.f": -3, "g": "hi"}


@dataclass
class Dummy:
    value: int
    ls: list[int]


def test_gcopy() -> None:
    obj = Dummy(5, [1])
    obj_scp = gcopy(obj, deep=False, value=6, ls=[0], num_copies=None)
    assert obj.value != obj_scp.value
    obj.ls[0] = 0
    assert obj.ls[0] == obj_scp.ls[0]

    obj_scp = gcopy(obj, deep=True, value=7, ls=[1], num_copies=None)
    assert obj.value != obj_scp.value
    assert obj.ls[0] != obj_scp.ls[0]

    obj_dcps = gcopy(obj, deep=True, value=6, ls=[0], num_copies=2)
    assert len(list(obj_dcps)) == 2
    for obj in obj_dcps:
        assert obj.value != obj_scp.value
        assert obj.ls[0] != obj_scp.ls[0]


def test_strenum() -> None:
    class _Things(StrEnum):
        POTATO = auto()
        ORANGE = auto()
        SPADE = auto()

    for thing in _Things:
        assert thing == thing.name.lower()
        assert thing.value == thing.name.lower()

        assert f"{thing}" == thing.name.lower()
        assert str(thing) == thing.name.lower()
        assert repr(thing) == f"<_Things.{thing.name}: '{thing.value}'>"


def test_strenum_custom_value() -> None:
    class _Cols(StrEnum):
        """These values are not valid variables names."""

        GENDER_MALE = "gender-male"
        MORE_THAN_50K = ">50K"

    assert _Cols.GENDER_MALE == "gender-male"
    assert _Cols.MORE_THAN_50K == ">50K"
    assert str(_Cols.GENDER_MALE) == "gender-male"
    assert str(_Cols.MORE_THAN_50K) == ">50K"

    def f(s: str) -> None:
        assert isinstance(s, str)

    f(_Cols.GENDER_MALE)
    f(_Cols.MORE_THAN_50K)


def test_adddict() -> None:
    d1 = AddDict({"foo": 1, "bar": 2})
    d2 = AddDict({"foo": 3, "bar": 4})
    d12 = d1 + d2
    assert sum([d1, d2]) == d12
    assert d12["foo"] == d1["foo"] + d2["foo"]
    assert d12["bar"] == d1["bar"] + d2["bar"]

    d3 = AddDict({"foo": [1], "bar": [2]})
    d4 = {"foo": [3, 4], "bar": [4]}
    d34 = d3 + d4
    assert d34["foo"] == d3["foo"] + d4["foo"]
    assert d34["bar"] == d3["bar"] + d4["bar"]
    with pytest.raises(TypeError):
        d1 += d3


@pytest.mark.parametrize(
    "seed, train, val, test",
    [
        (0, [6, 1, 0, 5, 3, 8, 7], [9, 2], [4]),
        (1, [1, 0, 9, 8, 6, 3, 7], [4, 2], [5]),
        (888, [6, 4, 9, 7, 0, 1, 5], [8, 2], [3]),
    ],
)
def test_reproducible_random_split(
    seed: int, train: list[int], val: list[int], test: list[int]
) -> None:
    LEN = 10
    splits = reproducible_random_split(LEN, props=[0.7, 0.2, 0.1], seed=seed)
    assert len(splits) == 3
    assert sum(len(split) for split in splits) == LEN
    assert splits[0] == train
    assert splits[1] == val
    assert splits[2] == test

    # Do the same thing again.
    splits = reproducible_random_split(LEN, props=[0.7, 0.2, 0.1], seed=seed)
    assert len(splits) == 3
    assert sum(len(split) for split in splits) == LEN
    assert splits[0] == train
    assert splits[1] == val
    assert splits[2] == test
