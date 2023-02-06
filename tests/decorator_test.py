"""Test decorators."""
from __future__ import annotations
from enum import Enum, auto
from typing import List, Union

import pytest

from ranzen import enum_name_str, implements, parsable


def test_implements() -> None:
    """Test the implements decorator."""

    class BaseClass:
        def func(self) -> None:
            """Do nothing."""

        def no_docstring(self) -> None:
            pass

    class CorrectImplementation(BaseClass):  # pyright: ignore
        @implements(BaseClass)
        def func(self) -> None:
            pass

    with pytest.raises(AssertionError):

        class IncorrectImplementation(BaseClass):  # pyright: ignore
            @implements(BaseClass)
            def wrong_func(self) -> None:
                pass

    class NoDocstringImpl(BaseClass):  # pyright: ignore
        @implements(BaseClass)
        def no_docstring(self) -> None:
            pass


def test_parsable() -> None:
    class Foo:  # pyright: ignore
        @parsable
        def __init__(self) -> None:
            ...


def test_parsable_valid() -> None:
    class Foo:  # pyright: ignore
        @parsable
        def __init__(self, a: int, b: Union[int, float], c: List[str]):  # pyright: ignore
            ...


def test_parsable_invalid_union() -> None:
    with pytest.raises(ValueError):

        class Foo:  # pyright: ignore
            @parsable
            def __init__(self, a: int, b: int | float, c: List[str]):  # pyright: ignore
                ...


@enum_name_str
class Stage(Enum):
    """An enum for the stage of model-development."""

    fit = auto()
    """fitting stage"""
    validate = auto()
    """validation stage"""
    test = auto()
    """testing stage"""


def test_enum_str() -> None:
    for stage in Stage:
        assert f"{stage}" == stage.name
        assert f"{stage!r}" == f"Stage.{stage.name}"
