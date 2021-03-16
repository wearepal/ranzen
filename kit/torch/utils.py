from __future__ import annotations
import time
from collections.abc import Iterable, Iterator
import random
from typing import TypeVar

import numpy as np
import torch

__all__ = ["count_parameters", "random_seed", "inf_generator", "Event"]


def count_parameters(model):
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_seed(seed_value, use_cuda) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


T = TypeVar("T")


def inf_generator(iterable: Iterable[T]) -> Iterator[T]:
    """Get DataLoaders in a single infinite loop.

    for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iter(iterable)
    # try to take one element to ensure that the iterator is not empty
    first_value = next(iterator, None)
    if first_value is not None:
        yield first_value
    else:
        raise RuntimeError("The given iterable is empty.")
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class Event():
    """Emulates torch.cuda.Event, but supports running on a CPU too.

    Examples:
    >>> from kit.torch import Event
    >>> start = Event()
    >>> end = Event()
    >>> start.record()
    >>> y = some_nn_module(x)
    >>> end.record()
    >>> print(start.elapsed_time(end))
    """

    def __init__(self):
        self.event_obj = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.time = 0

    def record(self):
        """Mark a time.

        Mimics torch.cuda.Event.
        """
        if torch.cuda.is_available():
            assert self.event_obj is not None
            self.event_obj.record()
        else:
            self.time = time.time()

    def elapsed_time(self, e: Event) -> int:
        """Measure difference between 2 times.

        Mimics torch.cuda.Event.
        """
        if not torch.cuda.is_available():
            return e.time - self.time
        assert self.event_obj is not None
        assert isinstance(e.event_obj, torch.cuda.Event)
        return self.event_obj.elapsed_time(e.event_obj)