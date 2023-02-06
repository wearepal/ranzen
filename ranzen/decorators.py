"""Decorator functions."""
from __future__ import annotations
from enum import Enum
from functools import partial
import inspect
from typing import (
    Any,
    Callable,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

from loguru import logger
import wrapt  # pyright: ignore

from ranzen.misc import some

__all__ = [
    "deprecated",
    "enum_name_str",
    "implements",
    "parsable",
]


_T = TypeVar("_T")
_W = TypeVar("_W", bound=Union[Callable[..., Any], Type[Any]])


class IdentityFunction(Protocol[_T]):
    def __call__(self, __x: _T) -> _T:
        ...


# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


@overload
def deprecated(
    wrapped: _W,
    /,
    *,
    version: str | None = ...,
    explanation: str | None = ...,
) -> _W:
    ...


@overload
def deprecated(
    wrapped: None = ...,
    /,
    *,
    version: str | None = ...,
    explanation: str | None = ...,
) -> IdentityFunction:
    ...


def deprecated(
    wrapped: _W | None = None,
    /,
    *,
    version: str | None = None,
    explanation: str | None = None,
) -> _W | IdentityFunction:
    """
    Decorator which can be used for indicating that a function/class is deprecated and going to be removed.
    Tracks down which function/class printed the warning and will print it only once per call.

    :param wrapped: Function/class to be marked as deprecated.
    :param version: Version in which the function/class will be removed..
    :param explanation: Additional explanation, e.g. "Please, ``use another_function`` instead." .

    :returns: Function/class wrapped with a deprecation warning.
    """

    if wrapped is None:
        return partial(deprecated, version=version, explanation=explanation)

    @wrapt.decorator
    def wrapper(wrapped: _W, *args: Any, **kwargs: Any) -> _W:  # pyright: ignore
        # Check if we already warned about the given function/class.
        if wrapped.__name__ not in _PRINTED_WARNING.keys():
            # Add to list so we won't log it again.
            _PRINTED_WARNING[wrapped.__name__] = True

            # Prepare the warning message.
            entity_name = "Class" if inspect.isclass(wrapped) else "Function"
            msg = f"{entity_name} '{wrapped.__name__}' is deprecated"

            # Optionally, add version and explanation.
            if some(version):
                msg = f"{msg} and will be removed in version {version}"

            msg = f"{msg}."
            if some(explanation):
                msg = f"{msg} {explanation}"

            # Display the deprecated warning.
            logger.warning(msg)

        # Call the function/initialise the class.
        return cast(_W, wrapped)

    return wrapper(wrapped)


_F = TypeVar("_F", bound=Callable[..., Any])


@deprecated(explanation="Use 'typing_extensions.override' instead.")
class implements:  # pylint: disable=invalid-name
    """Mark a function as implementing an interface.

    .. warning::
        This decorator is deprecated in favour of :function:`typing_extensions.override` instead.
    """

    def __init__(self, interface: type):
        """Instantiate the decorator.

        :param interface: the interface that is implemented
        """
        self.interface = interface

    def __call__(self, func: _F) -> _F:
        """Take a function and return it unchanged."""
        super_method = getattr(self.interface, func.__name__, None)
        assert super_method is not None, f"'{func.__name__}' does not exist in {self.interface}"
        return func


def parsable(func: _F) -> _F:
    """Mark an object's __init__ as parsable by Configen, so only pre-3.9 type annotations should be used."""
    assert func.__name__ == "__init__", "@parsable can only be used to decorate __init__."
    try:
        get_type_hints(func)
    except TypeError:
        raise ValueError(
            "the type annotations of this function are not automatically parseable."
        ) from None
    return func


E = TypeVar("E", bound=Enum)


def enum_name_str(enum_class: type[E]) -> type[E]:
    """Patch the __str__ method of an enum so that it returns the name."""
    # use the original __str__ method as __repr__
    enum_class.__repr__ = enum_class.__str__

    def __str__(self: Enum) -> str:
        return self.name.lower()

    enum_class.__str__ = __str__  # type: ignore
    return enum_class
