from __future__ import annotations
import copy
from enum import Enum
import functools
import operator
from typing import Any, Dict, Iterable, MutableMapping, TypeVar, Union, overload

from typing_extensions import Self

from ranzen.types import Addable

__all__ = [
    "AddDict",
    "StrEnum",
    "flatten_dict",
    "gcopy",
    "reduce_add",
    "str_to_enum",
]


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


try:
    # will be available in python 3.11
    from enum import StrEnum  # type: ignore
except ImportError:
    #
    # the following is copied straight from https://github.com/python/cpython/blob/3.11/Lib/enum.py
    #
    # DO NOT CHANGE THIS CODE!
    #
    from enum import Enum

    class ReprEnum(Enum):
        """
        Only changes the repr(), leaving str() and format() to the mixed-in type.
        """

    class StrEnum(str, ReprEnum):
        """
        Enum where members are also (and must be) strings
        """

        def __new__(cls, *values):
            "values must already be of type `str`"
            if len(values) > 3:
                raise TypeError("too many arguments for str(): %r" % (values,))
            if len(values) == 1:
                # it must be a string
                if not isinstance(values[0], str):
                    raise TypeError("%r is not a string" % (values[0],))
            if len(values) >= 2:
                # check that encoding argument is a string
                if not isinstance(values[1], str):
                    raise TypeError("encoding must be a string, not %r" % (values[1],))
            if len(values) == 3:
                # check that errors argument is a string
                if not isinstance(values[2], str):
                    raise TypeError("errors must be a string, not %r" % (values[2]))
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]):
            """
            Return the lower-cased version of the member name.
            """
            return name.lower()


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class AddDict(Dict[_KT, _VT], Addable):
    """
    Extension of the built-in dictionary class that supports use of the ``__add__`` operator for
    merging its values with those of other dictionaries. Note that, for the sake of simplicity,
    addition requires both dictionaries to be of the same generic type. Values that do not have an
    ``__add__`` operator defined, either in general or with respect to the other value, will be
    merged into a list

    :example:

    .. code-block:: python

        # Simple case of addition of integers.
        d1 = AddDict({"foo": 1, "bar": 2})
        d2 = {"foo": 3, "bar": 4}
        d1 + d2 # {'foo': 4, 'bar': 6}

        # Concatenation of lists
        d3 = AddDict({"foo": [1], "bar": [2]})
        d4 = {"foo": [3, 4], "bar": 4}
        d3 + d4 # {'foo': [1, 3, 4], 'bar': [2, 4]}
    """

    @overload
    def __add__(self: Self, other: int) -> Self:
        ...

    @overload
    def __add__(self: Self, other: dict[_KT, _VT]) -> AddDict[_KT, _VT | list[_VT]]:
        ...

    def __add__(
        self: Self,
        other: int | dict[_KT, _VT],
    ) -> Self | AddDict[_KT, _VT | list[_VT]]:
        # Allow ``other`` to be an integer, but specifying the identity function, for compatibility
        # with th 'no-default' version of``sum``.
        if isinstance(other, int):
            return self
        copy: AddDict[_KT, Union[_VT, list[_VT]]] = AddDict()
        copy.update(gcopy(self, deep=False))

        def _fallback(x1: _VT, x2: _VT) -> list[_VT]:
            if isinstance(x1, list):
                return x1 + [x2] 
            elif isinstance(x2, list):
                return x2 + [x1]
            return [x1, x2]

        for key_o, value_o in other.items():
            if key_o in self:
                value_s = self[key_o]
                if isinstance(value_s, Addable) and isinstance(value_o, Addable):
                    # Values are mutually addable.
                    try:
                        copy[key_o] = value_s + value_o
                    except TypeError:
                        copy[key_o] = _fallback(value_s, value_o)
                else:
                    copy[key_o] = _fallback(value_s, value_o)
            else:
                copy[key_o] = value_o
        return copy

    @overload
    def __radd__(self: Self, other: int) -> Self:
        ...

    @overload
    def __radd__(self: Self, other: dict[_KT, _VT]) -> AddDict[_KT, _VT | list[_VT]]:
        ...

    def __radd__(self: Self, other: int | dict[_KT, _VT]) -> Self | AddDict[_KT, _VT | list[_VT]]:
        return self + other


A = TypeVar("A", bound=Addable)


def reduce_add(sequence: Iterable[A]) -> A:
    """
    Sum an iterable using functools.reduce and operator.add rather than the built-in sum operator
    to bypass need to specify an initial value or have the elements ``__add__`` operator be
    compatible with integers (0 being the default initial value).

    :param sequence: An iterable of addable instances, all of the same type (invariant).
    :returns: The sum of all elements in ``__iterable``.
    """
    return functools.reduce(operator.add, sequence)
