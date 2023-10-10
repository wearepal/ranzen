"""Test decorators."""
from enum import Enum, auto
from typing_extensions import deprecated

import pytest

from ranzen import enum_name_str  # pyright: ignore


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
