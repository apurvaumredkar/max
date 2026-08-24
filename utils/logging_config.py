"""
Central logging setup. Everything goes to stdout, which systemd captures into the journal —
so `journalctl -u max-agent` (and the web UI's Logs tab, which streams the same source) is
the single place to look.

Call configure_logging() once at startup, then use get_logger(__name__) per module.
"""

import logging
import os
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-16s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level=None):
    """
    Install a stdout handler on the root logger. Idempotent — safe to call more than once.

    Level comes from the LOG_LEVEL env var (default INFO), so verbosity is changeable via
    secrets/.env without touching code.
    """
    global _configured
    if _configured:
        return logging.getLogger("max")

    level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; let them propagate to ours instead so the format
    # is uniform and there are no duplicate lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # httpx logs a line per request at INFO, which drowns out everything else during
    # streaming completions.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger("max")


def get_logger(name):
    """Get a module logger. Name is trimmed to its last segment to keep log lines narrow."""
    configure_logging()
    return logging.getLogger(f"max.{name.rsplit('.', 1)[-1]}")
