"""Functions for dealing with hydra."""
from __future__ import annotations
from collections.abc import MutableMapping
from dataclasses import asdict
from enum import Enum
import shlex
from typing import Any, Sequence

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

__all__ = ["flatten_dict", "as_pretty_dict", "reconstruct_cmd", "recursively_instantiate"]


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
    return shlex.join([program] + OmegaConf.to_container(args))


def recursively_instantiate(
    hydra_config: DictConfig, keys_to_exclude: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        str(k): instantiate(v, _convert_="partial")
        for k, v in hydra_config.items()
        if k not in ("_target_",) + tuple(keys_to_exclude)
    }
