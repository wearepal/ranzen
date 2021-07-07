"""Test decorators."""
from __future__ import annotations
from typing import List, Union

import pytest

from kit import implements
from kit.decorators import parsable


def test_implements() -> None:
    """Test the implements decorator."""

    class BaseClass:
        def func(self) -> None:
            """Do nothing."""

        def no_docstring(self) -> None:
            pass

    class CorrectImplementation(BaseClass):
        @implements(BaseClass)
        def func(self) -> None:
            pass

    with pytest.raises(AssertionError):

        class IncorrectImplementation(BaseClass):
            @implements(BaseClass)
            def wrong_func(self) -> None:
                pass

    # with pytest.raises(AssertionError):

    class NoDocstringImpl(BaseClass):
        @implements(BaseClass)
        def no_docstring(self) -> None:
            pass


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
