from __future__ import annotations
import copy
from enum import Enum
from typing import Any, MutableMapping, TypeVar, overload

__all__ = ["flatten_dict", "gcopy", "str_to_enum"]


def flatten_dict(
    d: MutableMapping[str, Any], *, parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`.

    :param d: Dictionary to be flattened.
    :param parent_key: Key-prefix (separated from the key with 'sep') to use for top-level
        keys of the flattened dictionary.
    :param sep: Character to separate the parent keys from the child keys with at each level with.

    :returns: Flattened dictionary with keys capturing the nesting path as ``parent_key.child_key``,
        where 'parent_key' is defined recursively, with base value 'parent_key' as specified in the
        function call.
    """
    items: list[tuple[str, Any]] = []
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
def gcopy(obj: T, *, deep: bool = True, num_copies: int, **kwargs: Any) -> list[T]:
    ...


def gcopy(
    obj: T, *, deep: bool = True, num_copies: int | None = None, **kwargs: Any
) -> T | list[T]:
    """Generalised (g) copy function.
    Allows for switching between deep and shallow copying within a single function
    as well as for the creation of multiple copies and for copying while simultaneously
    attribute-setting.

    :param obj: Object to be copied.
    :param deep: Whether to create deep (True) or shallow (False) copies.
    :param num_copies: Number of copies to create with 'None' being equivalent to 1.
    :param kwargs: Key-word arguments specifying a name of an attribute and the
        new value to set it to in the copies.

    :returns: A copy or list of copies (if num_copies > 1) of the object 'obj'.

    :raises AttributeError: If an attribute specified in ``kwargs`` doesn't exist.
    """
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


E = TypeVar("E", bound=Enum)


def str_to_enum(str_: str | E, *, enum: type[E]) -> E:
    """Convert a string to an enum based on name instead of value.
    If the string is not a valid name of a member of the target enum,
    an error will be raised.

    :param str_: String to be converted to an enum member of type ``enum``.
    :param enum: Enum class to convert ``str_`` to.

    :returns: The enum member of type ``enum`` with name ``str_``.

    :raises TypeError: if the given string is not a valid name of a member of the target enum
    """
    if isinstance(str_, enum):
        return str_
    try:
        return enum[str_]  # type: ignore
    except KeyError:
        valid_ls = [mem.name for mem in enum]
        raise TypeError(
            f"'{str_}' is not a valid option for enum '{enum.__name__}'; must be one of {valid_ls}."
        )
