from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
from unittest.mock import patch

from attr import dataclass

from kit.hydra import Option
from kit.hydra.relay import Relay


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


def test_relay(tmpdir: Path) -> None:
    args = ["", "A=foo", "B=bar"]
    with patch("sys.argv", args):
        options = dict(
            A=[Option(DummyOptionA, "foo"), DummyOptionB],
            B=[DummyOptionA, Option(DummyOptionB, "bar")],
        )
        for _ in range(2):
            DummyRelay.with_hydra(base_config_dir=tmpdir, **options)
        conf_dir = tmpdir / DummyRelay._config_dir_name()
        assert conf_dir.exists()
        assert (conf_dir / "configen" / DummyRelay._CONFIGEN_FILENAME).exists()
        for key in options.keys():
            assert (conf_dir / key).exists()
