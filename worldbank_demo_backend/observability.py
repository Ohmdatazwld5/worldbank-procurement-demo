"""Structured application/audit events and optional OpenTelemetry tracing."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Iterator


LOGGER = logging.getLogger("worldbank_demo")
_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    log_path = os.getenv("APP_LOG_PATH", "logs/application.jsonl")
    path = __import__("pathlib").Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False
    _LOGGING_CONFIGURED = True


def emit_event(event_type: str, **attributes: Any) -> dict[str, Any]:
    configure_logging()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        **attributes,
    }
    LOGGER.info(json.dumps(event, default=str, sort_keys=True))
    return event


@contextmanager
def trace_operation(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Emit consistent latency/error events; attach an OTel span when configured."""
    span = None
    try:
        from opentelemetry import trace

        span = trace.get_tracer("worldbank_demo").start_span(name)
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
    except ImportError:
        span = None
    started = time.perf_counter()
    outcome: dict[str, Any] = {"status": "ok"}
    try:
        yield outcome
    except Exception as exc:
        outcome["status"] = "error"
        outcome["error_type"] = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        emit_event(
            "operation",
            operation=name,
            duration_ms=duration_ms,
            **attributes,
            **outcome,
        )
        from .monitoring import _record_operation_histogram

        histogram = _record_operation_histogram()
        if histogram is not None:
            histogram.record(duration_ms, attributes={"operation": name})
        if span is not None:
            span.end()
