"""
The classes in this file define the configuration variables for SuperDuperDB,
which means that this file gets imported before alost anything else, and
canot contain any other imports from this project.
"""

import json
import os
import typing as t
from enum import Enum
from os.path import expanduser

from .jsonable import Factory, JSONable

_CONFIG_IMMUTABLE = True


class BaseConfigJSONable(JSONable):
    def force_set(self, name, value):
        """
        Forcefully setattr of BaseConfigJSONable instance
        """
        super().__setattr__(name, value)

    def __setattr__(self, name, value):
        if not _CONFIG_IMMUTABLE:
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f'Process attempted to set "{name}" attribute of immutable configuration '
            f'object {self}.'
        )


class Retry(BaseConfigJSONable):
    """
    Describes how to retry using the `tenacity` library

    :param stop_after_attempt: The number of attempts to make
    :param wait_max: The maximum time to wait between attempts
    :param wait_min: The minimum time to wait between attempts
    :param wait_multiplier: The multiplier for the wait time between attempts
    """

    stop_after_attempt: int = 2
    wait_max: float = 10.0
    wait_min: float = 4.0
    wait_multiplier: float = 1.0


class Model(BaseConfigJSONable):
    """
    Describes configuration of the AI model used by Knapsack.
    """
    provider: str = "sentence_transformers"
    id: str


class LogLevel(str, Enum):
    """
    Enumerate log severity level
    """
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    SUCCESS = "SUCCESS"
    WARN = 'WARN'
    ERROR = 'ERROR'
    EXCEPTION = 'EXCEPTION'


class LogType(str, Enum):
    """
    Enumerate the standard logs

    SYSTEM uses the systems STDOUT and STDERR for printing the logs.
    DEBUG, INFO, and WARN go to STDOUT.
    ERROR goes to STDERR.
    """
    SYSTEM = "SYSTEM"


class KnapsackConfig(BaseConfigJSONable):
    """
    The data class containing all configurable Knapsack values

    :param knapsack_dir: Dir for all Knapsack data.
    :param data_backend: The URI for the data backend
    :param vector_search: The configuration for the vector search {'in_memory', 'lance'}
    :param artifact_store: The URI for the artifact store
    :param metadata_store: The URI for the metadata store
    :param retries: Settings for retrying failed operations

    :param downloads_folder: Settings for downloading files

    :param fold_probability: The probability of validation fold

    :param log_level: The severity level of the logs
    :param logging_type: The type of logging to use
    """
    knapsack_dir: str = expanduser('~/knapsack')
    log_dir: str = expanduser('~/knapsack/logs/ks.log')

    log_level: LogLevel = LogLevel.INFO
    # logging_type: LogType = LogType.SYSTEM

    # dot_env: t.Optional[str] = None

    @property
    def comparables(self):
        """
        A dict of `self` excluding some defined attributes.
        """
        _dict = self.dict()
        list(map(_dict.pop, ('cluster', 'retries', 'downloads_folder')))
        return _dict
