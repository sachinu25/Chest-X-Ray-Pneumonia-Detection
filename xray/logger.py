"""
Centralized logging configuration.

Supports two modes controlled by the LOG_TARGET environment variable:

- **local** (default): Logs to both a local file (logs/{timestamp}/) and console.
  Ideal for development and local debugging.

- **cloud**: Logs structured JSON to stdout only. Designed for containerized
  environments (ECS Fargate, EC2 Docker) where CloudWatch or similar log drivers
  capture stdout automatically. No local file I/O — safe for ephemeral containers.

Usage:
    from xray.logger import logging

    # Set LOG_TARGET=cloud before import for production containers.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

from xray.constant.training_pipeline import TIMESTAMP

LOG_TARGET = os.environ.get("LOG_TARGET", "local").lower()

LOG_FORMAT = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# JSON formatter for cloud / container environments
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields:
        timestamp, level, name, message, environment
    CloudWatch, Datadog, and ELK all parse this natively.
    """

    def __init__(self):
        super().__init__()
        self.environment = os.environ.get("ENVIRONMENT", "production")

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "environment": self.environment,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Configure root logger based on target
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    """Set up the root logger handlers based on LOG_TARGET."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicates on re-import
    root.handlers.clear()

    if LOG_TARGET == "cloud":
        # Cloud mode — structured JSON to stdout only
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        root.addHandler(handler)
    else:
        # Local mode — file + console (original behaviour)
        logs_path = os.path.join(os.getcwd(), "logs", TIMESTAMP)
        os.makedirs(logs_path, exist_ok=True)
        log_file_path = os.path.join(logs_path, f"{TIMESTAMP}.log")

        root.addHandler(logging.FileHandler(log_file_path))
        root.addHandler(logging.StreamHandler(sys.stdout))
        for h in root.handlers:
            h.setFormatter(logging.Formatter(LOG_FORMAT))

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


_configure_logging()
