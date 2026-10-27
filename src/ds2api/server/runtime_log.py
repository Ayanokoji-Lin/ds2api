"""Runtime logging: console + rotating file + in-memory ring buffer.

Port of ``src/server/runtime_log.rs``. Three sinks:

* stderr, colourised only when attached to a TTY
* ``${DS_DATA_DIR}/logs/runtime.log``, 10 MB per file, 3 rotated copies
* a 2000-entry ring buffer that backs ``GET /admin/api/runtime-logs``

Log messages are kept in English, matching the upstream project's convention.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

RING_CAPACITY = 2000
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 3

_ring: deque[dict] | None = None
_handler: "RingHandler | None" = None
_installed = False

_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",
    "INFO": "\x1b[32m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;31m",
}
_RESET = "\x1b[0m"


class RingHandler(logging.Handler):
    """Keep the most recent records in memory for the admin API."""

    def __init__(self, capacity: int = RING_CAPACITY) -> None:
        super().__init__()
        self.buffer: deque[dict] = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{logging.Formatter().formatException(record.exc_info)}"
            self.buffer.append(
                {
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
                    )
                    + f".{int(record.msecs):03d}",
                    "level": record.levelname,
                    "target": record.name,
                    "message": message,
                }
            )
        except Exception:  # noqa: BLE001 - never let logging break the app
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not sys.stderr.isatty():
            return text
        color = _LEVEL_COLORS.get(record.levelname)
        return f"{color}{text}{_RESET}" if color else text


def install(data_dir: Path | str, level_name: str = "info") -> RingHandler:
    """Configure the root logger. Safe to call more than once."""
    global _ring, _handler, _installed
    if _installed and _handler is not None:
        return _handler

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ColorFormatter("[%(asctime)s %(levelname)-5s  %(name)s] %(message)s",
                                        datefmt="%Y-%m-%dT%H:%M:%S"))
    root.addHandler(console)

    log_path = Path(data_dir) / "logs" / "runtime.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)-5s  %(name)s] %(message)s",
                              datefmt="%Y-%m-%dT%H:%M:%S")
        )
        root.addHandler(file_handler)
    except OSError:
        root.warning("cannot open %s; file logging disabled", log_path)

    handler = RingHandler()
    root.addHandler(handler)

    # uvicorn keeps its own loggers; let them propagate to the root config.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    _handler = handler
    _ring = handler.buffer
    _installed = True
    return handler


def entries(offset: int = 0, limit: int = 100) -> list[dict]:
    if _ring is None:
        return []
    # Newest first, matching the dashboard's expectation.
    items = list(_ring)[::-1]
    return items[max(0, offset) : max(0, offset) + max(0, limit)]


def total() -> int:
    return len(_ring) if _ring is not None else 0


def reset() -> None:
    """Test helper: drop the installed handlers."""
    global _ring, _handler, _installed
    _ring = None
    _handler = None
    _installed = False
