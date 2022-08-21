from __future__ import annotations
from dataclasses import dataclass
from enum import auto

import pytest

from ranzen import AddDict, StrEnum, flatten_dict, gcopy


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
