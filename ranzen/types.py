from typing import TypeVar

from typing_extensions import Protocol, Self, runtime_checkable

__all__ = ["Sized", "Addable"]


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Sized(Protocol[T_co]):
    def __len__(self) -> int:
        ...


@runtime_checkable
class Addable(Protocol[T_co]):
    def __add__(self, other: Self, /) -> Self:
        ...
