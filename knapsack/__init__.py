# ruff: noqa: E402
from .base import config, configs, jsonable
from .base.logger import logger

ICON = '🎒'
ROOT = configs.ROOT
CFG = configs.CFG

from knapsack.knapsack import Knapsack

__version__ = '0.1.0'

__all__ = (
    'CFG',
    'ICON',
    'ROOT',
    'jsonable',
    'config',
    'Knapsack',
    'logger',
)
