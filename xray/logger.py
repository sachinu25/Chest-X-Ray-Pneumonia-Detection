"""
Centralized logging configuration.

Logs to both file (for persistence) and console (for development visibility).
Log files are stored in logs/{timestamp}/ with timestamped filenames.
"""

import logging
import os
import sys

from xray.constant.training_pipeline import TIMESTAMP

LOG_FILE: str = f"{TIMESTAMP}.log"

logs_path = os.path.join(os.getcwd(), "logs", TIMESTAMP)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

LOG_FORMAT = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        # File handler — persistent log storage
        logging.FileHandler(LOG_FILE_PATH),
        # Console handler — visible during development
        logging.StreamHandler(sys.stdout),
    ],
)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
