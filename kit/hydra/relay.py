from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum
from functools import lru_cache
import importlib
import inspect
import logging
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing_extensions import Final, final

from .utils import SchemaRegistration

__all__ = [
    "Relay",
    "Option",
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


@dataclass(init=False)
class Option:
    def __init__(self, class_: type[Any], name: str | None = None) -> None:
        self.class_ = class_
        self._name = name

    @property
    def name(self) -> str:
        if self._name is None:
            cls_name = self.class_.__name__
            if cls_name.endswith("Conf"):
                cls_name.rstrip("Conf")
            return _camel_to_snake(cls_name)
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:  # type: ignore
        self._name = name


class _SchemaImportInfo(NamedTuple):
    schema_name: str
    name: str
    module: ModuleType | Path


R = TypeVar("R", bound="Relay")


class Relay:
    """
    Abstract class for orchestrating hydra runs.

    This class does away with the hassle of needing to define config-stores, initialise
    config directories, and manually run configen on classes to convert them into valid schemas.
    Regular non-hydra compatible, classes can be passed to the `with_hydra` method and
    configen will be run on them automatically (if use_cached_confs=False or a cached version
    of the schemas can't be found), with the resulting schemas cached in the config directory.

    Subclasses must implement a 'run' method and will themselves be converted into the
    primary config classes used to initialise and validate hydra. With the launching of hydra,
    the subclasses will be instantiated and the run method called on the raw config.

    :example:

    >>>
    Relay.with_hydra(
        base_config_dir="conf",
        model=[Option(MoCoV2), Option(DINO)],
        datamodule=[Option(ColoredMNISTDataModule, "cmnist")],
    )

    """

    _CONFIG_NAME: ClassVar[str] = "config"
    _PRIMARY_SCHEMA_NAME: ClassVar[str] = "relay_schema"
    _CONFIGEN_FILENAME: ClassVar[str] = "conf.py"
    _logger: ClassVar[Optional[logging.Logger]] = None

    @classmethod
    def _get_logger(cls: type[R]) -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging.INFO)
            cls._logger = logger
        return cls._logger

    @classmethod
    def log(cls: type[R], msg: str) -> None:
        cls._get_logger().info(msg)

    @classmethod
    def _config_dir_name(cls: type[R]) -> str:
        return _camel_to_snake(cls.__name__)

    @final
    @classmethod
    def _init_yaml_files(
        cls: type[R], *, config_dir: Path, config_dict: dict[str, list[Any]]
    ) -> None:
        primary_conf_fp = (config_dir / cls._CONFIG_NAME).with_suffix(".yaml")
        primary_conf_exists = primary_conf_fp.exists()
        with primary_conf_fp.open("a+") as primary_conf:
            if not primary_conf_exists:
                cls.log(f"Initialising primary config file '{primary_conf.name}'.")

                primary_conf.write(f"defaults:")
                primary_conf.write(f"\n{YAML_INDENT}- {cls._PRIMARY_SCHEMA_NAME}")

            for group, group_options in config_dict.items():
                group_dir = config_dir / group
                if not group_dir.exists():
                    group_dir.mkdir()
                    default = "" if len(group_options) > 1 else group_options[0].name
                    primary_conf.write(f"\n{YAML_INDENT}- {group}: {default}")

                cls.log(f"Initialising group '{group}'")
                for option in group_options:
                    open((group_dir / "defaults").with_suffix(".yaml"), "a").close()
                    with (group_dir / option.name).with_suffix(".yaml").open("w") as schema_config:
                        schema_config.write(f"defaults:")
                        schema_config.write(f"\n{YAML_INDENT}- /schema/{group}: {option.name}")
                        schema_config.write(f"\n{YAML_INDENT}- defaults")

                        sig = inspect.signature(option.class_.__init__)
                        for name, param in sig.parameters.items():
                            if name in ("self", "args", "kwargs"):
                                continue
                            entry = f"{name}: "
                            default = param.default
                            if not default is param.empty:
                                default_str = _to_yaml_value(default)
                                if default_str is None:
                                    if isinstance(default, type):
                                        default_str = f"{default.__module__}.{default.__name__}"
                                    else:
                                        class_path = f"{default.__class__.__module__}.{default.__class__.__name__}"
                                        default_str = f"\n{YAML_INDENT}# _target_: {class_path}"
                                        default_sig = inspect.signature(default.__class__.__init__)
                                        for key in default_sig.parameters.keys():
                                            if key in ("self", "args", "kwargs"):
                                                continue
                                            default_str += f"\n{YAML_INDENT}# {key}:"
                                    entry = f"# {entry}{default_str}"
                                else:
                                    entry += f"{default_str}"
                            schema_config.write(f"\n{entry}")
                        cls.log(f"- Initialising config file '{schema_config.name}'.")

        cls.log(f"Finished initialising config directory initialised at '{config_dir}'")

    @classmethod
    def _module_to_fp(cls: type[R], module: ModuleType | str):
        if isinstance(module, ModuleType):
            module = module.__name__
        return module.replace(".", "/")

    @classmethod
    def _generate_conf(
        cls: type[R], output_dir: Path, *, module_class_dict: dict[str, List[str]]
    ) -> None:
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
    def _load_module_from_path(cls: type[R], filepath: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(  # type: ignore
            name=filepath.name, location=str(filepath)
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)
        sys.modules[filepath.name] = module
        return module

    @classmethod
    def _load_schemas(
        cls: type[R],
        config_dir: Path,
        *,
        use_cached_confs: bool = True,
        **options: list[type[Any] | Option],
    ) -> Tuple[type[Any], DefaultDict[str, List[Option]], DefaultDict[str, List[Option]]]:
        configen_dir = config_dir / "configen"
        primary_schema_fp = (
            configen_dir / cls._module_to_fp(cls.__module__) / cls._CONFIGEN_FILENAME
        )
        schemas_to_generate = defaultdict(list)
        if not use_cached_confs and configen_dir.exists():
            configen_dir.rmdir()
        if not primary_schema_fp.exists():
            schemas_to_generate[cls.__module__].append(cls.__name__)
        imported_schemas: DefaultDict[str, list[Option]] = defaultdict(list)
        schemas_to_import: DefaultDict[str, list[_SchemaImportInfo]] = defaultdict(list)
        schemas_to_init: DefaultDict[str, list[Option]] = defaultdict(list)

        for group, group_options in options.items():
            for option in group_options:
                if not isinstance(option, Option):
                    option = Option(class_=option)
                if not (config_dir / group / option.name).with_suffix(".yaml").exists():
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
                            imported_schemas[group].append(
                                replace(option, class_=schema)  # type: ignore
                            )
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
                module = info.module
                if isinstance(module, Path):
                    module = cls._load_module_from_path(module)

                imported_schemas[group].append(
                    Option(class_=getattr(module, info.schema_name), name=info.name)  # type: ignore
                )

        return primary_schema, imported_schemas, schemas_to_init

    @final
    @classmethod
    def _launch(
        cls: type[R],
        *,
        base_config_dir: Path | str,
        use_cached_confs: bool = True,
        **options: list[type[Any] | Option],
    ) -> None:
        base_config_dir = Path(base_config_dir)
        config_dir_name = cls._config_dir_name()
        config_dir = (base_config_dir / config_dir_name).expanduser().resolve()
        config_dir.mkdir(exist_ok=True, parents=True)

        primary_schema, schemas, schemas_to_init = cls._load_schemas(
            config_dir=config_dir, use_cached_confs=use_cached_confs, **options
        )
        # Initialise any missing yaml files
        if schemas_to_init:
            cls.log(
                f"One or more config files not found in config directory {config_dir}."
                "\nInitialising missing config files."
            )
            cls._init_yaml_files(config_dir=config_dir, config_dict=schemas_to_init)
            cls.log(f"Relaunch the relay, modifying the config files first if desired.")
            return

        sr = SchemaRegistration()
        sr.register(path=cls._PRIMARY_SCHEMA_NAME, config_class=primary_schema)
        for group, schema_ls in schemas.items():
            with sr.new_group(group_name=f"schema/{group}", target_path=f"{group}") as group:
                for info in schema_ls:
                    group.add_option(name=info.name, config_class=info.class_)

        # config_path only allows for relative paths; we need to resort to argv-manipulation
        # in order to set the config directory with an absolute path
        sys.argv.extend(["--config-dir", str(config_dir)])

        @hydra.main(config_path=None, config_name=cls._CONFIG_NAME)
        def launcher(cfg: Any) -> None:
            exp: R = instantiate(cfg, _recursive_=True)
            config_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, enum_to_str=True))
            exp.run(config_dict)

        launcher()

    @classmethod
    def with_hydra(
        cls: type[R],
        base_config_dir: Path | str,
        *,
        use_cached_confs: bool = True,
        **options: list[type[Any] | Option],
    ) -> None:
        """Run the relay with hydra."""
        cls._launch(base_config_dir=base_config_dir, use_cached_confs=use_cached_confs, **options)

    @abstractmethod
    def run(self, raw_config: dict[str, Any] | None = None) -> None:
        ...
