from palkit import flatten_dict


def test_flatten_dict() -> None:
    d = {"a": 3, "b": {"c": 7.0, "d": {"e": True}, "f": -3}, "g": "hi"}
    assert flatten_dict(d) == {"a": 3, "b.c": 7.0, "b.d.e": True, "b.f": -3, "g": "hi"}
