import socket
import uuid
from os.path import expanduser
from pathlib import Path
from sys import exc_info, stderr

from tqdm import tqdm
from loguru import logger

from knapsack.base.config import LogLevel

from .configs import CFG

# if CFG.logging_type == LogType.LOKI:  # Send logs to Loki
#     custom_handler = LokiLoggerHandler(
#         url=os.environ["LOKI_URI"],
#         labels={"application": "Test", "environment": "Develop"},
#         labelKeys={},
#         timeout=10,
#         defaultFormatter=LoguruFormatter(),
#     )

#     logger.configure(handlers=[{"sink": custom_handler, "serialize": True}])
# else:

# Replace default logger with a custom Knapsack format.
logger.remove()

# exception = exc_info()
# logger.opt(exception=exception).info("Logging exception traceback")

# Enrich logger with additional information.
# logger.configure(
#     extra={
#         "hostname": socket.gethostname(),
#         "session_id": str(uuid.uuid4()),
#     }
# )

fmt = (
    "<green> {time:YYYY-MMM-DD HH:mm:ss.SS}</green>"
    "| <level>{level: <8}</level> "
    "| <blue>{level: <8}</blue> "
#     "| <cyan>{extra[hostname]: <8}</cyan>"
#     "| <cyan>{extra[session_id]}</cyan>"
    "| <cyan>{name}</cyan>:<cyan>{line: <4}</cyan> "
    "| <level>{message}</level> "
    "| {exception}"
)

# DEBUG until WARNING are sent to ks.log.
logger.add(
    # lambda msg: tqdm.write(msg, end=""),
    Path(expanduser(CFG.log_dir)) / Path("ks.log"),
    format=fmt,
    level=CFG.log_level,
    filter=lambda record: record["level"].no < 40,
    backtrace=True,
    colorize=False,
    # rotation="500 MB",
)

# ERROR and above sent to ks_error.log
# https://loguru.readthedocs.io/en/stable/api/logger.html
logger.add(
    Path(expanduser(CFG.log_dir)) / Path("ks_error.log"),
    # stderr,
    format=fmt,
    level=LogLevel.ERROR,
    backtrace=True,
    diagnose=False,
    colorize=False,
    # rotation="500 MB",
)

# Set Multi-Key loggers
# Example: logging.info("param 1", "param 2", ..)
# @staticmethod
# def multikey_debug(msg: str, *args):
#     logger.opt(depth=1).debug(" ".join(map(str, (msg, *args))))
# 
# @staticmethod
# def multikey_info(msg: str, *args):
#     logger.opt(depth=1).info(" ".join(map(str, (msg, *args))))
# 
# @staticmethod
# def multikey_success(msg: str, *args):
#     logger.opt(depth=1).success(" ".join(map(str, (msg, *args))))
# 
# @staticmethod
# def multikey_warn(msg: str, *args):
#     logger.opt(depth=1).warning(" ".join(map(str, (msg, *args))))
# 
# @staticmethod
# def multikey_error(msg: str, *args):
#     logger.opt(depth=1).error(" ".join(map(str, (msg, *args))))
# 
# @staticmethod
# def multikey_exception(msg: str, *args, e=None):
#     logger.opt(depth=1, exception=e).error(" ".join(map(str, (msg, *args))))

# logger = logger

# debug = multikey_debug
# info = multikey_info
# success = multikey_success
# warn = multikey_warn
# error = multikey_error
# exception = multikey_exception
