import time

import pytest

from mantra.torch.utils import Event


def test_event() -> None:
    with Event() as event:
        time.sleep(0.10)
    pytest.approx(event.time, 0.10)
    print(event)
