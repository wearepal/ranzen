from __future__ import annotations
from typing import List, Union

import pytest

from kit.typing import parsable


def test_parsable() -> None:
    class Foo:
        @parsable
        def __init__(self) -> None:
            ...


def test_parsable_valid() -> None:
    class Foo:
        @parsable
        def __init__(self, a: int, b: Union[int, float], c: List[str]):
            ...


def test_parsable_invalid_union() -> None:
    with pytest.raises(ValueError):

        class Foo:
            @parsable
            def __init__(self, a: int, b: int | float, c: List[str]):
                ...


def test_parsable_invalid_list() -> None:
    with pytest.raises(ValueError):

        class Foo:
            @parsable
            def __init__(self, a: int, b: Union[int, float], c: list[str]):
                ...
