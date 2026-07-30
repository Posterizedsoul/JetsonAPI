"""In-memory ring buffer of recent log records, for the Logs page.

Deliberately not `docker logs`: that would mean mounting the docker socket into
the gateway, which hands the container root on the host. A bounded deque costs
nothing and shows the same application output.

Only the last MAX_RECORDS are kept — this is a live tail, not an archive. For
history use `docker compose logs`, which is unaffected by this.
"""

import logging
from collections import deque
from datetime import datetime, timezone

MAX_RECORDS = 800

_records: deque = deque(maxlen=MAX_RECORDS)

# uvicorn configures its own loggers with propagate=False, so attaching to the
# root logger alone would miss every request line.
_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "app")


class RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _records.append({
                "ts": datetime.fromtimestamp(record.created, timezone.utc),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            })
        except Exception:
            # A logging handler must never raise into the caller.
            pass


def install(level: int = logging.INFO) -> None:
    handler = RingHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    for name in _LOGGERS:
        logger = logging.getLogger(name)
        # Idempotent: reloads in dev must not stack handlers.
        if not any(isinstance(h, RingHandler) for h in logger.handlers):
            logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)


def records(level: str | None = None, contains: str | None = None,
            limit: int = 200) -> list[dict]:
    """Newest first, optionally filtered."""
    out = list(_records)
    if level and level != "ALL":
        # A level filter means "this and worse", which is what you want when
        # hunting a problem.
        floor = logging.getLevelName(level)
        if isinstance(floor, int):
            out = [r for r in out
                   if logging.getLevelName(r["level"]) >= floor]
    if contains:
        needle = contains.lower()
        out = [r for r in out if needle in r["message"].lower()
               or needle in r["logger"].lower()]
    out.reverse()
    return out[:limit]


def clear() -> None:
    _records.clear()
