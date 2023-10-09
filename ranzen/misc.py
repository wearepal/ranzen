from __future__ import annotations
from collections.abc import Mapping
import copy
from enum import Enum, auto
import functools
import operator
import sys
from typing import Any, Dict, Iterable, Type, TypeVar, overload
from typing_extensions import Self, TypeGuard

from ranzen.types import Addable

__all__ = [
    "AddDict",
    "StrEnum",
    "default_if_none",
    "flatten_dict",
    "gcopy",
    "reduce_add",
    "some",
    "str_to_enum",
    "unwrap_or",
    "Stage",
    "Split",
]


def flatten_dict(d: Mapping[str, Any], *, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
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
        if isinstance(v, Mapping):
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
    obj_cp = copy.deepcopy(obj) if deep else copy.copy(obj)
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


if sys.version_info >= (3, 11):
    # will be available in python 3.11
    from enum import StrEnum
else:
    #
    # the following is copied straight from https://github.com/python/cpython/blob/3.11/Lib/enum.py
    #
    # DO NOT CHANGE THIS CODE!
    #
    class ReprEnum(Enum):
        """
        Only changes the repr(), leaving str() and format() to the mixed-in type.
        """

    _S = TypeVar("_S", bound="StrEnum")

    class StrEnum(str, ReprEnum):
        """
        Enum where members are also (and must be) strings
        """

        _value_: str

        def __new__(cls: Type[_S], *values: str) -> _S:
            "values must already be of type `str`"
            if len(values) > 3:
                raise TypeError("too many arguments for str(): %r" % (values,))
            if len(values) == 1:
                # it must be a string
                if not isinstance(values[0], str):  # pyright: ignore
                    raise TypeError("%r is not a string" % (values[0],))
            if len(values) >= 2:
                # check that encoding argument is a string
                if not isinstance(values[1], str):  # pyright: ignore
                    raise TypeError("encoding must be a string, not %r" % (values[1],))
            if len(values) == 3:
                # check that errors argument is a string
                if not isinstance(values[2], str):  # pyright: ignore
                    raise TypeError("errors must be a string, not %r" % (values[2]))
            value = str(*values)
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return str.__str__(self)

        def _generate_next_value_(  # type: ignore
            name: str,
            start: int,
            count: int,
            last_values: list[Any],
        ) -> str:
            """
            Return the lower-cased version of the member name.
            """
            return name.lower()


_KT = TypeVar("_KT")
_VT = TypeVar("_VT", bound=Addable)
_VT2 = TypeVar("_VT2", bound=Addable)


class AddDict(Dict[_KT, _VT], Addable):
    """
    Extension of the built-in dictionary class that supports the use of the ``__add__`` operator for
    key-wise addition.

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
    def __add__(self, other: int) -> Self:
        ...

    @overload
    def __add__(self, other: dict[_KT, _VT2]) -> AddDict[_KT, _VT | _VT2]:
        ...

    def __add__(self, other: int | dict[_KT, _VT2]) -> Self | AddDict[_KT, _VT | _VT2]:
        # Allow ``other`` to be an integer, but specifying the identity function, for compatibility
        # with th 'no-default' version of``sum``.
        if isinstance(other, int):
            return self
        copy: AddDict[_KT, _VT | _VT2] = AddDict()
        copy.update(gcopy(self, deep=False))

        for key_o, value_o in other.items():
            if not isinstance(value_o, Addable):
                raise TypeError(f"Value of type '{type(value_o)}' is not addable.")
            if key_o in self:
                value_s = self[key_o]
                if not isinstance(value_s, Addable):
                    raise TypeError(f"Value of type '{type(value_s)}' is not addable.")
                try:
                    # Values are mutually addable (but not necessarily of the same type).
                    copy[key_o] = value_s + value_o
                except TypeError as e:
                    msg = (
                        f"Values of type '{type(value_s)}' and '{type(value_o)}' for key "
                        f"'{key_o}' are not mutuablly addable."
                    )
                    raise TypeError(msg) from e
            else:
                copy[key_o] = value_o
        return copy

    @overload
    def __radd__(self, other: int) -> Self:
        ...

    @overload
    def __radd__(self, other: dict[_KT, _VT2]) -> AddDict[_KT, _VT | _VT2]:
        ...

    def __radd__(self, other: int | dict[_KT, _VT2]) -> Self | AddDict[_KT, _VT | _VT2]:
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


def some(value: T | None, /) -> TypeGuard[T]:
    """
    Returns ``True`` if the input is **not** ``None``
    (that is, if the ``Optional`` monad contains some value).

    :param value: Value to be checked.
    :returns: ``True`` if ``value`` is **not** ``None`` else ``False``.
    """
    return value is not None


def unwrap_or(value: T | None, /, *, default: T) -> T:
    """
    Returns the input if the input is **not** None else the specified
    ``default`` value.

    :param value: Input to be unwrapped and returned if not ``None``.
    :param default: Default value to use if ``value`` is ``None``.
    :returns: ``default`` if ``value`` is ``None`` otherwise ``value``.
    """
    return default if value is None else value


default_if_none = unwrap_or


class Stage(StrEnum):
    FIT = auto()
    VALIDATE = auto()
    TEST = auto()


class Split(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
