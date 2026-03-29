import json
import logging
import sys
from datetime import UTC, datetime


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def configure_logging(log_level: str = "INFO") -> logging.Logger:
    numeric = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {log_level!r}")

    logger = logging.getLogger("mara")
    logger.setLevel(numeric)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)

    return logger
