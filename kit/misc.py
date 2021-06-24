from __future__ import annotations
from typing import Any, Iterator, MutableMapping, TypeVar, overload

__all__ = ["flatten_dict", "copy"]


def flatten_dict(
    d: MutableMapping[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


T = TypeVar("T")


@overload
def copy(obj: T, deep: bool = True, num_copies: None = ..., **kwargs: Any) -> T:
    ...


@overload
def copy(obj: T, deep: bool = True, num_copies: int = ..., **kwargs: Any) -> Iterator[T]:
    ...


def copy(
    obj: T, deep: bool = True, num_copies: int | None = None, **kwargs: Any
) -> T | Iterator[T]:
    if num_copies is not None:
        for _ in range(num_copies):
            yield copy(obj=obj, deep=deep, num_copies=None, kwargs=kwargs)
    copy_fn = copy.deepcopy if deep else copy.copy
    obj_cp = copy_fn(obj)
    for attr, value in kwargs.items():
        if not hasattr(obj_cp, attr):
            raise AttributeError(f"Object of type {type(obj_cp)} has no attribute {attr}.")
        setattr(obj_cp, attr, value)
    return obj_cp
