from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
from unittest.mock import patch

from attr import dataclass
import pytest

from ranzen.hydra import Option
from ranzen.hydra.relay import Relay


class DummyOptionA:
    def __init__(self, name: str = "a", value: Union[int, float] = 7) -> None:
        self.name = name
        self.value = value


class DummyOptionB:
    def __init__(self, name: str = "b", value: Union[int, float] = 5) -> None:
        self.name = name
        self.value = value


@dataclass
class DummyRelay(Relay):

    A: DummyOptionA
    B: DummyOptionB

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        print(raw_config)
        print("running")


@pytest.mark.parametrize("clear_cache", [True, False])
def test_relay(tmpdir: Path, clear_cache: bool) -> None:
    args = ["", "A=foo", "B=bar"]
    with patch("sys.argv", args):
        options = dict(
            A=[Option(DummyOptionA, "foo"), DummyOptionB],
            B=[DummyOptionA, Option(DummyOptionB, "bar")],
        )
        for _ in range(2):
            DummyRelay.with_hydra(root=tmpdir, clear_cache=clear_cache, **options)
        conf_dir = tmpdir / DummyRelay._config_dir_name()  # pylint: disable=protected-access
        assert conf_dir.exists()
        assert (
            conf_dir
            / "configen"
            / "relay_test"
            / DummyRelay._CONFIGEN_FILENAME  # pylint: disable=protected-access
        ).exists()
        for key in options.keys():
            assert (conf_dir / key).exists()
