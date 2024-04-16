import os
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import tomllib
import yaml

from knapsack.base import config_utils
from knapsack.base.config import KnapsackConfig

File = t.Union[Path, str]

# The top-level directory of the project
ROOT = Path(__file__).parents[2]

# The default prefix used for config environment variables
PREFIX = 'KNAPSACK_'

# The name of the environment variable used to read the config files.
# This value needs to be read before all the other config values are.
FILES_NAME = 'CONFIG_FILES'

# The base name of the configs file
cfg_path = os.getenv('KNAPSACK_CONFIG')
if cfg_path:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise ValueError("KNAPSACK_CONFIG env is not a valid file.")
CONFIG_FILE = cfg_path if cfg_path else Path(os.path.expanduser('knapsack.toml'))

_LOCAL_CONFIG = Path(CONFIG_FILE)


@dataclass(frozen=True)
class ConfigSettings:
    """
    A class that reads a Pydantic class from config files and environment variables.

    :param cls: The Pydantic class to read.
    :param default_files: The default config files to read.
    :param prefix: The prefix to use for environment variables.
    :param environ: The environment variables to read from.
    """

    cls: t.Type
    default_files: t.Union[t.Sequence[Path], Path]
    prefix: str
    environ: t.Optional[t.Dict] = None
    base_config: t.Optional[KnapsackConfig] = None

    @cached_property
    def config(self) -> t.Any:
        """Read a Pydantic class"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                content = f.read()
                kwargs = tomllib.loads(content)
        else:
            kwargs = {}

        # if self.base_config:
        #     parent = self.base_config.dict()
        # else:
        #     parent = self.cls().dict()

        env = dict(os.environ if self.environ is None else self.environ)
        env = config_utils.environ_to_config_dict('KNAPSACK_', env)

        kwargs = config_utils.combine_configs((kwargs, env))
        return self.cls(**kwargs)


def build_config(cfg: t.Optional[KnapsackConfig] = None) -> KnapsackConfig:
    """
    Build the config object from the environment variables and config files.
    """
    CONFIG = ConfigSettings(KnapsackConfig, CONFIG_FILE, PREFIX)
    return CONFIG.config


CFG = build_config()
