"""Structured JSON-lines logging with a run id on every line (NFR-08) and secret redaction (NFR-04).

Usage:
    run_id = new_run_id()
    configure_logging(run_id)
    log = get_logger("data")
    log.info("downloaded", extra={"fields": {"ticker": "SPY", "bars": 5000}})

Every line is one JSON object: ts (UTC, ISO 8601), level, logger, run_id, msg and any
`fields`. Values that look like API keys, and the values of environment variables whose
names contain KEY, SECRET, TOKEN or PASSWORD, are replaced by "[REDACTED]" before output.
"""

import json
import logging
import os
import re
import secrets
import sys
from datetime import UTC, datetime
from typing import IO, Any

ROOT_LOGGER = "atu2"
REDACTED = "[REDACTED]"
_SECRET_ENV_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
_MIN_SECRET_LEN = 8
# Alpaca key ids (PK/AK/CK + 16+ upper-case alphanumerics) and long base64-like secrets.
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:PK|AK|CK)[A-Z0-9]{16,}\b"),
    re.compile(r"(?i)(?:secret|api[_-]?key|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{16,}"),
)


def new_run_id() -> str:
    """Run id: UTC timestamp plus 8 random hex characters, e.g. 20261126T101500Z-1a2b3c4d."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def _secret_env_values() -> list[str]:
    vals = []
    for name, value in os.environ.items():
        if any(m in name.upper() for m in _SECRET_ENV_MARKERS) and len(value) >= _MIN_SECRET_LEN:
            vals.append(value)
    return sorted(vals, key=len, reverse=True)


def redact(text: str) -> str:
    """Removes secret env-var values and key-like strings from `text`."""
    for value in _secret_env_values():
        text = text.replace(value, REDACTED)
    for pat in KEY_PATTERNS:
        text = pat.sub(REDACTED, text)
    return text


class JsonFormatter(logging.Formatter):
    """Formats a record as one redacted JSON line."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        """Returns the record as a JSON line with secrets removed."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": self.run_id,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                payload.setdefault(str(k), v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


def configure_logging(run_id: str, stream: IO[str] | None = None, level: int = logging.INFO) -> logging.Logger:
    """Sends every atu2.* logger to `stream` (default stderr) as JSON lines tagged with `run_id`."""
    root = logging.getLogger(ROOT_LOGGER)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter(run_id))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Logger named atu2.<name>; configure_logging must be called once per run."""
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
