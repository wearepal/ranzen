"""Test decorators."""
from __future__ import annotations
from enum import Enum, auto
from typing import List, Union

import pytest

from ranzen import deprecated, enum_name_str, parsable


@pytest.mark.parametrize("explanation", [None, "All things that have form eventually decay."])
@pytest.mark.parametrize("version", [None, "4.2"])
def test_deprecated(explanation: str | None, version: str | None) -> None:
    @deprecated(explanation=explanation, version=version)
    class Foo:  # pyright: ignore
        def __init__(self) -> None:
            ...

    @deprecated(explanation=explanation, version=version)
    def foo() -> None:  # pyright: ignore
        ...


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
