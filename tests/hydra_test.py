import dataclasses
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generic, TypeVar

from omegaconf import MISSING, DictConfig, MissingMandatoryValue, OmegaConf
import pytest

from ranzen.hydra import prepare_for_logging, register_hydra_config


def test_config_no_default() -> None:
    """This isn't so much wrong as just clumsy."""

    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class Config:
        dm: DataModule

    options = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)

    options = {"dm": {"base": DataModule}}
    register_hydra_config(Config, options)

    @dataclass
    class UnrelatedClass:
        root: Path

    options = {"dm": {"base": UnrelatedClass}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


def test_config_any() -> None:
    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class Config:
        dm: Any

    # we're assuming that the only reason you want to use Any is that
    # you want to use variants
    options = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)

    options = {"dm": {"base": DataModule}}
    register_hydra_config(Config, options)

    class NotDC:
        x: int

    options = {"dm": {"base": NotDC}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


def test_config_base_class() -> None:
    @dataclass
    class DataModule:
        root: Path

    @dataclass
    class CMnist(DataModule):
        colorize: bool

    @dataclass
    class CelebA(DataModule):
        target: str

    @dataclass
    class Config:
        dm: DataModule

    options = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)

    options = {"dm": {"cmnist": CMnist, "celeba": CelebA}}
    register_hydra_config(Config, options)

    @dataclass
    class NotSubclass:
        root: Path

    options = {"dm": {"base": NotSubclass}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


def test_config_any_with_default() -> None:
    """An Any field with default is completely out."""

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        model: Any = dataclasses.field(default_factory=Model)

    options = {}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)

    options = {"model": {"base": Model}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


def test_config_with_default() -> None:
    """A normal field with a default should not have variants."""

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        model: Model = dataclasses.field(default_factory=Model)

    options = {}
    register_hydra_config(Config, options)

    options = {"model": {"base": Model}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


T = TypeVar("T")


def test_config_generic() -> None:
    class Base(Generic[T]): ...

    @dataclass
    class DataModule(Base):
        root: Path

    @dataclass
    class Config:
        dm: DataModule

    options = {"dm": {"asdf": DataModule}}
    with pytest.raises(ValueError):
        register_hydra_config(Config, options)


def test_logging_dict() -> None:
    class TrainingType(Enum):
        iter = auto()
        epoch = auto()

    @dataclass
    class DataModule:
        root: Path = MISSING

    @dataclass
    class Model:
        layers: int = 1

    @dataclass
    class Config:
        dm: DataModule = dataclasses.field(default_factory=DataModule)
        model: Model = dataclasses.field(default_factory=Model)
        train: TrainingType = TrainingType.iter

    hydra_config: DictConfig = OmegaConf.structured(Config)
    hydra_config.model.layers = 3

    with pytest.raises(MissingMandatoryValue):  # `dm.root` is missing
        logging_dict = prepare_for_logging(hydra_config)

    hydra_config.dm.root = "."
    logging_dict = prepare_for_logging(hydra_config)

    assert logging_dict == {
        "dm/DataModule": {"root": Path()},
        "model/Model": {"layers": 3},
        "train": "iter",
    }
