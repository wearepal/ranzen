from __future__ import annotations
import copy
from enum import Enum
from typing import Any, MutableMapping, TypeVar, overload

__all__ = ["flatten_dict", "gcopy", "str_to_enum"]


def flatten_dict(
    d: MutableMapping[str, Any], *, parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


T = TypeVar("T")


@overload
def gcopy(obj: T, *, deep: bool = True, num_copies: None = ..., **kwargs: Any) -> T:
    ...


@overload
def gcopy(obj: T, *, deep: bool = True, num_copies: int = ..., **kwargs: Any) -> list[T]:
    ...


def gcopy(
    obj: T, *, deep: bool = True, num_copies: int | None = None, **kwargs: Any
) -> T | list[T]:
    if num_copies is not None:
        return [gcopy(obj=obj, deep=deep, num_copies=None, **kwargs) for _ in range(num_copies)]
    copy_fn = copy.deepcopy if deep else copy.copy
    obj_cp = copy_fn(obj)
    for attr, value in kwargs.items():
        if not hasattr(obj_cp, attr):
            raise AttributeError(
                f"Object of type '{type(obj_cp).__name__}' has no attribute '{attr}'."
            )
        setattr(obj_cp, attr, value)
    return obj_cp


T_co = TypeVar("T_co", covariant=False, bound=Enum)


def str_to_enum(str_: str, *, enum: type[T_co]) -> T_co:
    try:
        return enum[str_]  # type: ignore
    except KeyError:
        valid_ls = [mem.name for mem in enum]
        raise TypeError(
            f"'{str_}' is not a valid option for enum '{enum.__name__}'; must be one of {valid_ls}."
        )
