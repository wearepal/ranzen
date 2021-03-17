import time

from kit.torch.utils import Event
import pytest


def test_event():
    with Event() as event:
        time.sleep(0.10)
    pytest.approx(event.time, 0.10)
    print(event)
