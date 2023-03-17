"""Test decorators."""
from __future__ import annotations
from enum import Enum, auto
from typing import List, Union
from typing_extensions import deprecated

import pytest

from ranzen import enum_name_str, parsable  # pyright: ignore


@pytest.mark.parametrize("explanation", ["All things that have form eventually decay."])
def test_deprecated(explanation: str) -> None:
    @deprecated(explanation)
    class Foo:
        def __init__(self) -> None:
            ...

    with pytest.deprecated_call():
        Foo()  # pyright: ignore

    @deprecated(explanation)
    def foo() -> None:
        ...

    with pytest.deprecated_call():
        foo()  # pyright: ignore


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


def test_enum_str() -> None:
    with pytest.deprecated_call():

        @enum_name_str  # pyright: ignore
        class Stage(Enum):
            """An enum for the stage of model-development."""

            fit = auto()
            """fitting stage"""
            validate = auto()
            """validation stage"""
            test = auto()
            """testing stage"""

    for stage in Stage:
        assert f"{stage}" == stage.name
        assert f"{stage!r}" == f"Stage.{stage.name}"
