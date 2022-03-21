"""Functions for dealing with hydra."""
from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
import shlex
from typing import Any, Iterator, Sequence

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "GroupRegistration",
    "SchemaRegistration",
    "as_pretty_dict",
    "reconstruct_cmd",
    "recursively_instantiate",
]


def _clean_up_dict(obj: Any) -> Any:
    """Convert enums to strings and filter out _target_."""
    if isinstance(obj, MutableMapping):
        return {key: _clean_up_dict(value) for key, value in obj.items() if key != "_target_"}
    elif isinstance(obj, Enum):
        return str(f"{obj.name}")
    elif OmegaConf.is_config(obj):  # hydra stores lists as omegaconf.ListConfig, so we convert here
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    return obj


def as_pretty_dict(data_class: object) -> dict:
    """Convert dataclass to a pretty dictionary."""
    return _clean_up_dict(asdict(data_class))


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program."""
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = internal_config.overrides.task
    return _join([program] + OmegaConf.to_container(args))  # type: ignore[operator]


def _join(split_command: list[str]) -> str:
    """Concatenate the tokens of the list split_command and return a string."""
    return " ".join(shlex.quote(arg) for arg in split_command)


def recursively_instantiate(
    hydra_config: DictConfig, *, keys_to_exclude: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        str(k): instantiate(v, _convert_="partial")
        for k, v in hydra_config.items()
        if k not in ("_target_",) + tuple(keys_to_exclude)
    }


class SchemaRegistration:
    """Register hydra schemas.

    :example:

    >>> sr = SchemaRegistration()
    >>> sr.register(Config, path="experiment_schema")
    >>> sr.register(TrainerConf, path="trainer/trainer_schema")
    >>>
    >>> with sr.new_group("schema/data", target_path="data") as group:
    >>>    group.add_option(CelebaDataConf, name="celeba")
    >>>    group.add_option(WaterbirdsDataConf, name="waterbirds")
    """

    def __init__(self) -> None:
        self._cs = ConfigStore.instance()

    def register(self, config_class: type, *, path: str) -> None:
        """Register a schema."""
        if "." in path:
            raise ValueError(f"Separate path with '/' and not '.': {path}")

        parts = path.split("/")
        name = parts[-1]
        package = ".".join(parts[:-1])
        self._cs.store(name=name, node=config_class, package=package)

    @contextmanager
    def new_group(self, group_name: str, *, target_path: str) -> Iterator[GroupRegistration]:
        """Return a context manager for a new group."""
        package = target_path.replace("/", ".")
        yield GroupRegistration(self._cs, group_name=group_name, package=package)


class GroupRegistration:
    """Helper for registering a group in hydra."""

    def __init__(self, cs: ConfigStore, *, group_name: str, package: str):
        self._cs = cs
        self._group_name = group_name
        self._package = package

    def add_option(self, config_class: type, *, name: str) -> None:
        """Register a schema as an option for this group."""
        self._cs.store(group=self._group_name, name=name, node=config_class, package=self._package)
