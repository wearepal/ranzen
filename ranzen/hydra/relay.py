from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum
from functools import lru_cache
import importlib.util
import inspect
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    Final,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    final,
)
from typing_extensions import Self, TypeAlias

import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .utils import SchemaRegistration

__all__ = [
    "Relay",
    "Option",
    "Options",
]

YAML_INDENT: Final[str] = "  "


@lru_cache(maxsize=32)
def _camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _to_yaml_value(default: Any, *, indent_level: int = 0) -> str | None:
    str_ = None
    if default is None:
        str_ = "null"
    elif isinstance(default, str):
        str_ = f"'{default}'"
    elif isinstance(default, bool):
        str_ = str(default).lower()
    elif isinstance(default, Enum):
        str_ = default.name
    elif isinstance(default, (float, int)):
        str_ = str(default)
    elif isinstance(default, (tuple, list)):
        str_ = ""
        indent_level += 1
        for elem in default:
            elem_str = _to_yaml_value(elem, indent_level=indent_level)
            if elem_str is None:
                return None
            str_ += f"\n{YAML_INDENT * indent_level}- {elem_str}"
    elif isinstance(default, dict):
        str_ = ""
        indent_level += 1
        for key, value in default.items():
            value_str = _to_yaml_value(value, indent_level=indent_level)
            if value_str is None:
                return None
            str_ += f"\n{YAML_INDENT * indent_level} {key}: {value_str}"
    return str_


T = TypeVar("T", covariant=True)


@dataclass(init=False, unsafe_hash=True)
class Option(Generic[T]):
    """Configuration option."""

    def __init__(self, class_: type[T], name: str | None = None) -> None:
        self.class_ = class_
        self._name = name

    @property
    def name(self) -> str:
        """Name of the option."""
        if self._name is None:
            cls_name = self.class_.__name__
            if cls_name.endswith("Conf"):
                cls_name.rstrip("Conf")
            return _camel_to_snake(cls_name)
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name


class _SchemaImportInfo(NamedTuple):
    schema_name: str
    name: str
    module: ModuleType | Path


Options: TypeAlias = Union[
    Option, Type[T], Sequence[Type[T]], Sequence[Option[T]], Sequence[Union[Type[T], Option[T]]]
]


class Relay:
    """
    Abstract class for orchestrating runs with :mod:`hydra`.

    This class does away with the hassle of needing to define config-stores, initialise
    config directories, and manually run neoconfigen on classes to convert them into valid schemas.
    Regular non-hydra compatible, classes can be passed to  :meth:`with_hydra` and neoconfigen
    will be run on them automatically (if ``clear_cache=True`` or a cached version of the schemas
    can't be found), with the resulting schemas cached in the config directory.

    Subclasses must implement a :meth:`run` method and will themselves be converted into the
    primary config classes used to initialise and validate hydra. With the launching of hydra,
    the subclasses will be instantiated and the run method called on the raw config.

    :example:
        >>> Relay.with_hydra(
        >>>     root="conf",
        >>>     model=[Option(MoCoV2), DINO],
        >>>     datamodule=[Option(ColoredMNISTDataModule, "cmnist")],
        >>> )

    """

    _CONFIG_NAME: ClassVar[str] = "config"
    _PRIMARY_SCHEMA_NAME: ClassVar[str] = "relay_schema"
    _CONFIGEN_FILENAME: ClassVar[str] = "conf.py"
    _logger: ClassVar[Optional[logging.Logger]] = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.INFO)
            cls._logger = logger
        return cls._logger

    @classmethod
    def _log(cls, msg: str) -> None:
        cls._get_logger().info(msg)

    @classmethod
    def _config_dir_name(cls) -> str:
        return _camel_to_snake(cls.__name__)

    @final
    @classmethod
    def _init_yaml_files(cls, *, config_dir: Path, config_dict: dict[str, list[Any]]) -> None:
        primary_conf_fp = (config_dir / cls._CONFIG_NAME).with_suffix(".yaml")
        primary_conf_exists = primary_conf_fp.exists()
        with primary_conf_fp.open("a+") as primary_conf:
            if not primary_conf_exists:
                cls._log(f"Initialising primary config file '{primary_conf.name}'.")

                primary_conf.write("---\ndefaults:")
                primary_conf.write(f"\n{YAML_INDENT}- {cls._PRIMARY_SCHEMA_NAME}")
                primary_conf.write(f"\n{YAML_INDENT}- _self_")

            for group, group_options in config_dict.items():
                group_dir = config_dir / group
                if not group_dir.exists():
                    group_dir.mkdir()
                    default = "" if len(group_options) > 1 else f"{group_options[0].name}/default"
                    primary_conf.write(f"\n{YAML_INDENT}- {group}: {default}")

                cls._log(f"Initialising group '{group}'.")
                for option in group_options:
                    option_dir = group_dir / option.name
                    option_dir.mkdir(exist_ok=True)
                    schema_config = (option_dir / "default").with_suffix(".yaml")
                    with schema_config.open("w") as file:
                        alias = option_dir.with_suffix(".yaml")
                        if not alias.exists():
                            os.symlink(src=schema_config, dst=alias)
                        file.write("---\ndefaults:")
                        file.write(f"\n{YAML_INDENT}- /schema/{group}: {option.name}")
                        file.write(f"\n{YAML_INDENT}- _self_")

                        sig = inspect.signature(option.class_.__init__)
                        for name, param in sig.parameters.items():
                            # Skip self/args/kwargs
                            if name == "self" or (
                                param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                            ):
                                continue
                            entry = f"{name}: "
                            if (default := param.default) is not param.empty:
                                default_str = _to_yaml_value(default)
                                if default_str is None:
                                    entry = f"# {entry}???"
                                else:
                                    entry += f"{default_str}"
                            file.write(f"\n{entry}")
                        cls._log(f"- Initialising config file '{file.name}'.")

        cls._log(f"Finished initialising config directory initialised at '{config_dir}'")

    @classmethod
    def _module_to_fp(cls, module: ModuleType | str) -> str:
        if isinstance(module, ModuleType):
            module = module.__name__
        return module.replace(".", "/")

    @classmethod
    def _generate_conf(cls, output_dir: Path, *, module_class_dict: dict[str, List[str]]) -> None:
        from configen.config import ConfigenConf, ModuleConf  # type: ignore
        from configen.configen import generate_module  # type: ignore

        cfg = ConfigenConf(
            output_dir=str(output_dir),
            module_path_pattern=f"{cls._CONFIGEN_FILENAME}",
            modules=[],
            header="",
        )
        for module, classes in module_class_dict.items():
            module_conf = ModuleConf(name=module, classes=classes)
            code = generate_module(cfg=cfg, module=module_conf)
            output_dir.mkdir(parents=True, exist_ok=True)
            conf_dir = output_dir / cls._module_to_fp(module)
            conf_dir.mkdir(parents=True)
            conf_file = conf_dir / cls._CONFIGEN_FILENAME
            with conf_file.open("a+") as file:
                file.write(code)

    @classmethod
    def _load_module_from_path(cls, filepath: Path) -> ModuleType:
        import sys

        spec = importlib.util.spec_from_file_location(name="", location=str(filepath))
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[""] = module
        return module

    @classmethod
    def _load_schemas(
        cls,
        config_dir: Path,
        *,
        clear_cache: bool = False,
        **options: Options,
    ) -> Tuple[type[Any], DefaultDict[str, List[Option]], DefaultDict[str, List[Option]]]:
        configen_dir = config_dir / "configen"
        primary_schema_fp = (
            configen_dir / cls._module_to_fp(cls.__module__) / cls._CONFIGEN_FILENAME
        )
        schemas_to_generate = defaultdict(list)
        # Clear any cached schemas by deleting the configen directory
        if clear_cache and configen_dir.exists():
            shutil.rmtree(configen_dir)
        if not primary_schema_fp.exists():
            schemas_to_generate[cls.__module__].append(cls.__name__)
        imported_schemas: DefaultDict[str, list[Option]] = defaultdict(list)
        schemas_to_import: DefaultDict[str, list[_SchemaImportInfo]] = defaultdict(list)
        schemas_to_init: DefaultDict[str, list[Option]] = defaultdict(list)

        for group, group_options in options.items():
            if not isinstance(group_options, Sequence):
                group_options = [group_options]
            for option in group_options:
                if not isinstance(option, Option):
                    option = Option(class_=option)
                option_dir = config_dir / group / option.name
                if not (option_dir / "default").with_suffix(".yaml").exists():
                    schemas_to_init[group].append(option)
                cls_name = option.class_.__name__
                if (not is_dataclass(option.class_)) or (not cls_name.endswith("Conf")):
                    schema_name = f"{cls_name}Conf"
                    schema_missing = False
                    # Load the primary schema
                    secondary_schema_fp = (
                        configen_dir
                        / cls._module_to_fp(option.class_.__module__)
                        / cls._CONFIGEN_FILENAME
                    )
                    module = (
                        cls._load_module_from_path(secondary_schema_fp)
                        if secondary_schema_fp.exists()
                        else None
                    )
                    if module is None:
                        schema_missing = True
                    else:
                        schema = getattr(module, schema_name, None)
                        if schema is None:
                            schema_missing = True
                        else:
                            imported_schemas[group].append(replace(option, class_=schema))
                    if schema_missing:
                        schemas_to_generate[option.class_.__module__].append(cls_name)
                    import_info = _SchemaImportInfo(
                        schema_name=schema_name,
                        name=option.name,
                        module=secondary_schema_fp if module is None else module,
                    )
                    schemas_to_import[group].append(import_info)
                else:
                    imported_schemas[group].append(option)

        # Generate any confs with configen that have yet to be generated
        if schemas_to_generate:
            cls._generate_conf(output_dir=configen_dir, module_class_dict=schemas_to_generate)
        # Load the primary schema
        module = cls._load_module_from_path(primary_schema_fp)
        primary_schema = getattr(module, cls.__name__ + "Conf")
        # Load the sub-schemas
        for group, info_ls in schemas_to_import.items():
            for info in info_ls:
                module = cls._load_module_from_path(m) if isinstance(m := info.module, Path) else m

                schema = getattr(module, info.schema_name)
                # TODO: figure out why the below kludge (ostensibly) solves the issue of failed
                # attribute-retrieval during unpickling when using a paralllielising hydra
                # launcher and implement a more graceful solution.
                schema.__module__ = "__main__"
                imported_schemas[group].append(Option(class_=schema, name=info.name))

        return primary_schema, imported_schemas, schemas_to_init

    @final
    @classmethod
    def _launch(
        cls,
        *,
        root: Path | str,
        clear_cache: bool = False,
        instantiate_recursively: bool = True,
        **options: Options,
    ) -> None:
        root = Path(root)
        config_dir_name = cls._config_dir_name()
        config_dir = (root / config_dir_name).expanduser().resolve()
        config_dir.mkdir(exist_ok=True, parents=True)

        primary_schema, schemas, schemas_to_init = cls._load_schemas(
            config_dir=config_dir, clear_cache=clear_cache, **options
        )
        # Initialise any missing yaml files
        if schemas_to_init:
            cls._log(
                f"One or more config files not found in config directory '{config_dir}'."
                "\nInitialising missing config files."
            )
            cls._init_yaml_files(config_dir=config_dir, config_dict=schemas_to_init)
            cls._log(f"Relaunch the relay, modifying the config files first if desired.")
            return

        sr = SchemaRegistration()
        sr.register(path=cls._PRIMARY_SCHEMA_NAME, config_class=primary_schema)
        for group, schema_ls in schemas.items():
            with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group_:
                for info in schema_ls:
                    group_.add_option(name=info.name, config_class=info.class_)

        # config_path only allows for relative paths; we need to resort to construct a
        # searchpath plugin on-the-fly in order to set the config directory with an absolute path

        class RelayPlugin(SearchPathPlugin):
            def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
                search_path.prepend(provider=cls.__name__, path=str(config_dir.resolve()))

        Plugins().plugin_type_to_subclass_list[SearchPathPlugin].append(RelayPlugin)

        @hydra.main(config_path=None, config_name=cls._CONFIG_NAME, version_base=None)
        def launcher(cfg: Any, /) -> Any:
            relay: Self = instantiate(cfg, _recursive_=instantiate_recursively)
            config_dict = cast(
                Dict[str, Any],
                OmegaConf.to_container(cfg, throw_on_missing=True, enum_to_str=False, resolve=True),
            )
            return relay.run(config_dict)

        launcher()

    @classmethod
    def with_hydra(
        cls,
        root: Path | str,
        *,
        clear_cache: bool = False,
        instantiate_recursively: bool = True,
        **options: Options,
    ) -> None:
        """Run the relay with hydra.

        :param root: Root directory to look for the config directory in.
        :param clear_cache: Whether to clear the cached schemas and generate the schemas anew with
            neoconfigen.

        :param instantiate_recursively: Whether to recursively instantiate the relay instance.
        :param options: Option or sequence of options (value) to register for each group (key).
            If an option is a type or is an :class:`Option` with :attr:`Option.name` as ``None``,
            then a name will be generated based on the class name and used to register the option,
            else, the specified value for :attr:`Option.name` will be used.
        """
        cls._launch(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            **options,
        )

    @abstractmethod
    def run(self, raw_config: dict[str, Any] | None = None) -> Any:
        """Run the relay.
        :param raw_config: Dictionary containing the configuration used to instantiate the relay.
        """
        ...
