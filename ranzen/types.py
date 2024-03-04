from collections.abc import Mapping, Sequence
from dataclasses import Field
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeGuard,
    TypeVar,
    get_type_hints,
    is_typeddict,
    runtime_checkable,
)

__all__ = ["Addable", "DataclassInstance", "Sized", "SizedDataset", "Subset", "is_td_instance"]

T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Sized(Protocol[T_co]):
    def __len__(self) -> int: ...


@runtime_checkable
class Addable(Protocol[T_contra, T_co]):
    def __add__(self, other: T_contra, /) -> T_co: ...


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


TD = TypeVar("TD", bound=Mapping)


def is_td_instance(dict_: dict[str, Any], cls_: type[TD], *, strict: bool = False) -> TypeGuard[TD]:
    """``isinstance`` check for typed dictionaries.

    Returns ``True` if a dictionary conforms to a target ``TypedDict`` class.
    When ``strict` is ``False`` 'conforms' means that ``dict_`` contains all
    keys defined in ``cls_`` with values of the correct type -- additional keys
    are permitted. When ``True``, ``dict_`` must contain those keys **only**
    defined by ``cls_``.

    :param dict_: Dictionary to be checked.
    :param cls_: ``TypedDict`` class to compare ``dict_` against.

    :param strict: Whether to invalidate a ``dict_`` for containing additional
        keys not defined in ``cls_``

    :returns: ``True`` if ``dict_`` conforms to the target ``TypedDict`` class, ``cls_``,
        and ``False`` otherwise.
    """
    assert is_typeddict(cls_), "cls_ must be a TypedDict class."
    hints = get_type_hints(cls_)
    if strict and (len(dict_) != len(hints)):
        return False
    for key, type_ in hints.items():
        if (key not in dict_) or (not isinstance(dict_[key], type_)):
            return False
    return True


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class SizedDataset(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...

    def __len__(self) -> int: ...


D = TypeVar("D", bound=SizedDataset)


class Subset(Generic[D]):
    r"""
    Subset of a dataset at specified indices.

    :param dataset: The whole Dataset.
    :param indices: Indices in the whole set selected for subset.
    """

    dataset: D
    indices: Sequence[int]

    def __init__(self, dataset: D, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)
