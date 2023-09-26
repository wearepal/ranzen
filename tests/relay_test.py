from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol
from typing_extensions import Self
from unittest.mock import patch

import pytest

from ranzen.hydra.relay import Option, Options, Relay


class DummyOption(Protocol):
    name: str

    @property
    def value(self) -> int | str:
        ...


@dataclass
class DummyOptionA:
    name: str = "a"
    value: int = 7


@dataclass
class DummyOptionB:
    name: str = "b"
    value: str = "5"


@dataclass
class DummyRelay(Relay):
    attr1: DummyOption
    attr2: DummyOption

    @classmethod
    def with_hydra(
        cls: type[Self],
        root: Path | str,
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = True,
        attr1: Options[DummyOption],
        attr2: Options[DummyOption],
        **options: Option,
    ) -> None:
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            attr1=attr1,
            attr2=attr2,
        )

    def run(self, raw_config: Optional[Dict[str, Any]] = None) -> None:
        print(raw_config)
        print("running")


@pytest.mark.parametrize("clear_cache", [True, False])
@pytest.mark.parametrize("instantiate_recursively", [True, False])
def test_relay(tmpdir: Path, clear_cache: bool, instantiate_recursively: bool) -> None:
    args = ["", "attr1=foo", "attr2=bar"]
    with patch("sys.argv", args):
        ops1: Sequence[type[DummyOption] | Option[DummyOption]] = [
            Option(DummyOptionA, "foo"),
            DummyOptionB,
        ]
        ops2: Sequence[type[DummyOption] | Option[DummyOption]] = [
            DummyOptionA,
            Option(DummyOptionB, "bar"),
        ]
        options = {"attr1": ops1, "attr2": ops2}
        for _ in range(2):
            DummyRelay.with_hydra(
                root=tmpdir,
                clear_cache=clear_cache,
                instantiate_recursively=instantiate_recursively,
                attr1=ops1,
                attr2=ops2,
            )
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
