"""
Logging Levels:
    DEBUG: Detailed Internal State
    INFO: Normal Pipeline Progress
    WARNING: Recoverable Anomalies
    ERROR: Failed Operations
    CRITICAL: Abort Immediately

Log Structure:
    timestamp
    level
    module
    message
    optional metadata(dict)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import Any


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Define the logging function and return the logger.

    Args:
        name (str): The name of the logger.
        log_dir (str): The path where logs should be stored.
        level (int): Logging Severity Level

    Returns:
        Logging.Logger: Returns a defined logger."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt="%(message)s")

    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_event(logger: logging.Logger, level: int, message: str, **metadata: Any):
    """
    Logs events

    Args:
        logger (logging.Logger): Logging Function,
        level (int): Severity Level,
        message (str): Message to Log,
        **metadata (Any): keyword argument representing all metadata

    Returns:
        None
    """

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "message": message,
        "metadata": metadata,
    }

    logger.log(level, json.dumps(payload))
