"""Logging: a rotating app-event logger (`log`) and the JSONL audit writer (`_log`)."""

import json
import logging
import logging.handlers
import threading

from config import LOG_DIR, LOG_FILE

LOG_DIR.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
_handler_file = logging.handlers.RotatingFileHandler(
    LOG_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5
)
_handler_file.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_handler_file])
log = logging.getLogger("chiller")

_log_lock = threading.Lock()


def _log(entry: dict):
    """Append one structured event to the JSONL audit trail (grades/feedback/corrections)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry)
    with _log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
