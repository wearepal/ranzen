from __future__ import annotations
from dataclasses import dataclass

from kit import flatten_dict, gcopy


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
