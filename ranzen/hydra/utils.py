"""Functions for dealing with hydra."""

from collections.abc import Iterator, MutableMapping, Sequence
from contextlib import contextmanager
import dataclasses
from dataclasses import MISSING, asdict, is_dataclass
from enum import Enum
import shlex
from typing import Any, Final, cast, get_args, get_type_hints
from typing_extensions import deprecated

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ranzen.types import DataclassInstance

__all__ = [
    "GroupRegistration",
    "SchemaRegistration",
    "as_pretty_dict",
    "reconstruct_cmd",
    "recursively_instantiate",
    "prepare_for_logging",
    "register_hydra_config",
]

NEED: Final = "there should be"
IF: Final = "if an entry has"


def _clean_up_dict(obj: Any) -> Any:
    """Convert enums to strings and filter out _target_."""
    if isinstance(obj, MutableMapping):
        return {key: _clean_up_dict(value) for key, value in obj.items() if key != "_target_"}
    elif isinstance(obj, Enum):
        return str(f"{obj.name}")
    elif OmegaConf.is_config(obj):  # hydra stores lists as omegaconf.ListConfig, so we convert here
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    return obj


def as_pretty_dict(data_class: DataclassInstance) -> dict:
    """Convert dataclass to a pretty dictionary."""
    return _clean_up_dict(asdict(data_class))


def reconstruct_cmd() -> str:
    """Reconstruct the python command that was used to start this program."""
    internal_config = HydraConfig.get()
    program = internal_config.job.name + ".py"
    args = internal_config.overrides.task
    return shlex.join([program] + OmegaConf.to_container(args))  # type: ignore[operator]


@deprecated("Use _recursive_=True instead.")
def recursively_instantiate(
    hydra_config: DictConfig, *, keys_to_exclude: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        str(k): instantiate(v, _convert_="partial")
        for k, v in hydra_config.items()
        if k not in ("_target_",) + tuple(keys_to_exclude)
    }


class GroupRegistration:
    """Helper for registering a group in hydra."""

    def __init__(self, cs: ConfigStore, *, group_name: str, package: str):
        self._cs = cs
        self._group_name = group_name
        self._package = package

    def add_option(self, config_class: type, *, name: str) -> None:
        """Register a schema as an option for this group."""
        self._cs.store(group=self._group_name, name=name, node=config_class, package=self._package)


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


def prepare_for_logging(hydra_config: DictConfig, *, enum_to_str: bool = True) -> dict[str, Any]:
    """Takes a hydra config dict and makes it prettier for logging.

    Things this function does: turn enums to strings, resolve any references, mark entries with
    their type.
    """
    raw_config = OmegaConf.to_container(
        hydra_config, throw_on_missing=True, enum_to_str=enum_to_str, resolve=True
    )
    assert isinstance(raw_config, dict)
    raw_config = cast(dict[str, Any], raw_config)
    return {
        f"{key}/{OmegaConf.get_type(dict_).__name__}"  # type: ignore
        if isinstance(dict_ := hydra_config[key], DictConfig)
        else key: value
        for key, value in raw_config.items()
    }


def register_hydra_config(
    main_cls: type, groups: dict[str, dict[str, type]], schema_name: str = "config_schema"
) -> None:
    """Check the given config and store everything in the ConfigStore.

    This function performs two tasks: 1) make the necessary calls to `ConfigStore`
    and 2) run some checks over the given config and if there are problems, try to give a nice
    error message.

    :param main_cls: The main config class; can be dataclass or attrs.
    :param groups: A dictionary that defines all the variants. The keys of top level of the
        dictionary should corresponds to the group names, and the keys in the nested dictionaries
        should correspond to the names of the options.
    :param schema_name: Name of the main schema. This name has to appear in the defaults list in the
        main config file.
    :raises ValueError: If the config is malformed in some way.
    :raises RuntimeError: If hydra itself is throwing an error.

    :example:
        .. code-block:: python

            @dataclass
            class DataModule:
                root: Path = Path()

            @dataclass
            class LinearModel:
                dim: int = 256

            @dataclass
            class CNNModel:
                kernel: int = 3

            @dataclass
            class Config:
                dm: DataModule = dataclasses.field(default_factory=DataModule)
                model: Any

            groups = {"model": {"linear": LinearModel, "cnn": CNNModel}}
            register_hydra_config(Config, groups)
    """
    assert isinstance(main_cls, type), "`main_cls` has to be a type."
    if not is_dataclass(main_cls):
        raise ValueError(f"The config class {main_cls.__name__} should be a dataclass.")
    entries = dataclasses.fields(main_cls)
    try:
        types = get_type_hints(main_cls)
    except NameError as exc:
        raise ValueError(
            f"Can't resolve type hints from the config class: `{main_cls.__name__}`."
        ) from exc

    for entry in entries:
        typ = types[entry.name]
        if typ == Any:
            if (group := groups.get(entry.name)) is not None:
                for var_name, var_class in group.items():
                    if not is_dataclass(var_class):
                        raise ValueError(
                            f"All variants should be dataclasses: type `{var_class.__name__}` "
                            f"of variant `{entry.name}={var_name}` is not a dataclass."
                        )
            else:
                raise ValueError(f"{IF} type `Any`, {NEED} variants: `{entry.name}`")
            if entry.default is not MISSING or entry.default_factory is not MISSING:
                raise ValueError(f"{IF} type `Any`, {NEED} no default value: `{entry.name}`")
        else:
            if is_dataclass(typ):
                if entry.default is MISSING and entry.default_factory is MISSING:
                    if (group := groups.get(entry.name)) is not None:
                        for var_name, var_class in group.items():
                            if not issubclass(var_class, typ):  # type: ignore
                                typ_name = typ.__name__  # type: ignore
                                raise ValueError(
                                    f"All variants should be subclasses of their entry's type: type"
                                    f" `{var_class.__name__}` of variant `{entry.name}={var_name}` "
                                    f"is not a subclass of `{typ_name}`."
                                )
                    else:
                        raise ValueError(
                            f"{IF} a dataclass type, "
                            f"{NEED} a default value or registered variants: `{entry.name}`. "
                            "You can specify a default value with `field(default_factory=...)`."
                        )
                else:
                    if entry.name in groups:
                        raise ValueError(
                            f"Can't have both a default value and variants: `{entry.name}`."
                        )
            elif entry.name in groups:
                raise ValueError(
                    f"Entry `{entry.name}` has registered variants, but its type annotation, "
                    f"`{getattr(typ, '__name__', str(typ))}`, is not a dataclass. "
                    "(Note that unions of dataclasses are not allowed either.) "
                    "You can always use `Any` for the type annotation."
                )

    cs = ConfigStore.instance()
    cs.store(node=main_cls, name=schema_name)
    for group, entries in groups.items():
        for var_name, var_type in entries.items():
            if (bases := getattr(var_type, "__orig_bases__", None)) is not None:
                if len(bases) > 0 and len(get_args(bases[0])) > 0:
                    raise ValueError(
                        f"Can't register a dataclass with generic base class: `{var_type.__name__}`"
                        f" with base class `{bases[0].__name__}`."
                    )
            try:
                cs.store(node=var_type, name=var_name, group=group)
            except Exception as exc:
                raise RuntimeError(f"{main_cls=}, {var_type=}, {var_name=}, {group=}") from exc
