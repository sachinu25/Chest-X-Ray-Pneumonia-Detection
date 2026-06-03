"""
Structured logging configuration for the Streamlit dashboard.

Logs to both console (INFO) and a rotating file (DEBUG) at logs/streamlit.log.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Ensure log directory exists
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "streamlit.log")


def get_logger(name: str = "streamlit_app") -> logging.Logger:
    """
    Get or create a named logger with console and file handlers.

    Args:
        name: Logger name (default: 'streamlit_app').

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO level)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (DEBUG level, rotating at 5 MB, keep 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
