"""Logging estruturado em JSON (padrão radioskysims), autocontido."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formata registros como JSON estruturado."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            log_obj.update(record.msg)
        else:
            log_obj["event"] = record.getMessage()
        if record.args and isinstance(record.args, dict):
            log_obj.update(record.args)
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False, default=str)


class StructuredLogger:
    """Logger que aceita kwargs para logging estruturado."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        msg_dict = {"event": event}
        msg_dict.update(kwargs)
        self._logger.log(level, msg_dict)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        msg_dict = {"event": event}
        msg_dict.update(kwargs)
        self._logger.exception(msg_dict, exc_info=True)


def get_logger(name: str) -> StructuredLogger:
    """Obtém um logger estruturado para um módulo."""
    return StructuredLogger(logging.getLogger(name))


def configure_logging(log_level: str = "INFO") -> None:
    """Configura o logging raiz com formato JSON em stdout."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("georinex").setLevel(logging.ERROR)
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


def log_http_event(
    logger: StructuredLogger,
    event: str,
    method: str,
    url: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    **kwargs: Any,
) -> None:
    """Registra evento HTTP em formato estruturado."""
    logger.info(
        event,
        category="http",
        method=method,
        url=url,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_io_event(
    logger: StructuredLogger,
    event: str,
    operation: str,
    path: str,
    bytes_count: int | None = None,
    **kwargs: Any,
) -> None:
    """Registra evento de IO em formato estruturado."""
    logger.info(
        event,
        category="io",
        operation=operation,
        path=path,
        bytes_count=bytes_count,
        **kwargs,
    )
