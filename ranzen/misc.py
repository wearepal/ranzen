from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
import copy
from enum import Enum, auto
import functools
import operator
import sys
from typing import Any, Literal, TypeGuard, TypeVar, overload
from typing_extensions import Self

import numpy as np

from ranzen.types import Addable, SizedDataset, Subset

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

__all__ = [
    "AddDict",
    "Split",
    "Stage",
    "StrEnum",
    "default_if_none",
    "flatten_dict",
    "gcopy",
    "prop_random_split",
    "reduce_add",
    "some",
    "str_to_enum",
    "unwrap_or",
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
def gcopy(obj: T, *, deep: bool = True, num_copies: None = ..., **kwargs: Any) -> T: ...


@overload
def gcopy(obj: T, *, deep: bool = True, num_copies: int, **kwargs: Any) -> list[T]: ...


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
    :param \\**kwargs: Key-word arguments specifying a name of an attribute and the
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


_KT = TypeVar("_KT")
_VT = TypeVar("_VT", bound=Addable)
_VT2 = TypeVar("_VT2", bound=Addable)


class AddDict(dict[_KT, _VT], Addable[int | dict, dict]):
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
    def __add__(self, other: int) -> Self: ...

    @overload
    def __add__(self, other: dict[_KT, _VT2]) -> AddDict[_KT, _VT | _VT2]: ...

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
    def __radd__(self, other: int) -> Self: ...

    @overload
    def __radd__(self, other: dict[_KT, _VT2]) -> AddDict[_KT, _VT | _VT2]: ...

    def __radd__(self, other: int | dict[_KT, _VT2]) -> Self | AddDict[_KT, _VT | _VT2]:
        # Calling `__add__` directly because with the `+` syntax, pyright complains for some reason.
        return self.__add__(other)


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


D = TypeVar("D", bound=SizedDataset)


@overload
def prop_random_split(
    dataset_or_size: D,
    *,
    props: Sequence[float] | float,
    as_indices: Literal[False] = ...,
    seed: int | None = ...,
    reproducible: bool = ...,
) -> list[Subset[D]]: ...


@overload
def prop_random_split(
    dataset_or_size: SizedDataset,
    *,
    props: Sequence[float] | float,
    as_indices: Literal[True],
    seed: int | None = ...,
    reproducible: bool = ...,
) -> list[list[int]]: ...


@overload
def prop_random_split(
    dataset_or_size: int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = ...,
    seed: int | None = ...,
    reproducible: bool = ...,
) -> list[list[int]]: ...


@overload
def prop_random_split(
    dataset_or_size: D | int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = ...,
    seed: int | None = ...,
    reproducible: bool = ...,
) -> list[Subset[D]] | list[list[int]]: ...


def prop_random_split(
    dataset_or_size: D | int,
    *,
    props: Sequence[float] | float,
    as_indices: bool = False,
    seed: int | None = None,
    reproducible: bool = False,
) -> list[Subset[D]] | list[list[int]]:
    """Splits a dataset based on proportions rather than on absolute sizes

    :param dataset_or_size: Dataset or size (length) of the dataset to split.
    :param props: The fractional size of each subset into which to randomly split the data.
        Elements must be non-negative and sum to 1 or less; if less then the size of the final
        split will be computed by complement.

    :param as_indices: If ``True`` the raw indices are returned instead of subsets constructed
        from them when `dataset_or_len` is a dataset. This means that when `dataset_or_len`
        corresponds to the length of a dataset, this argument has no effect and
        the function always returns the split indices.

    :param seed: The PRNG used for determining the random splits.

    :param reproducible: If ``True``, use a generator which is reproducible across machines,
        operating systems, and Python versions.

    :returns: Random subsets of the data of the requested proportions.

    :raises ValueError: If the dataset does not have a ``__len__`` method or sum(props) > 1.
    """
    if isinstance(dataset_or_size, int):
        len_ = dataset_or_size
    else:
        if not hasattr(dataset_or_size, "__len__"):
            raise ValueError(
                "Split proportions can only be computed for datasets with __len__ defined."
            )
        len_ = len(dataset_or_size)

    if isinstance(props, (float, int)):
        props = [props]
    sum_ = np.sum(props)
    if (sum_ > 1.0) or any(prop < 0 for prop in props):
        raise ValueError("Values for 'props` must be positive and sum to 1 or less.")
    section_sizes = [round(prop * len_) for prop in props]
    if (current_len := sum(section_sizes)) < len_:
        section_sizes.append(len_ - current_len)

    if reproducible:
        if seed is None:
            raise ValueError("Must specify seed for reproducible split.")
        # MT19937 isn't the best random number generator, but it's reproducible, so we're using it.
        generator = np.random.Generator(np.random.MT19937(seed))
    else:
        generator = np.random.default_rng(seed)
    indices = np.arange(sum(section_sizes))
    generator.shuffle(indices)  # Shuffle the indices in-place.

    splits = [
        indices[offset - length : offset].tolist()
        for offset, length in zip(np.cumsum(section_sizes), section_sizes)
    ]

    if as_indices or isinstance(dataset_or_size, int):
        return splits
    return [Subset(dataset_or_size, indices=split) for split in splits]
