"""Structured logs (NFR-08): JSON lines with a run id on every line; no secrets (NFR-04)."""

import io
import json
import logging
import re

from atu2.core.logging import REDACTED, configure_logging, get_logger, new_run_id, redact


def _fake_key_id() -> str:
    # built at runtime so no key-shaped literal sits in the repository
    return "PK" + "Q7" * 9


def test_every_line_is_json_with_the_run_id():
    run_id = new_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run_id)
    buf = io.StringIO()
    configure_logging(run_id, stream=buf)
    log = get_logger("data")
    log.info("downloaded", extra={"fields": {"ticker": "SPY", "bars": 5000}})
    log.warning("gap found")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    get_logger("other").debug("not shown at INFO")
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert all(r["run_id"] == run_id for r in records)
    assert records[0]["ticker"] == "SPY" and records[0]["bars"] == 5000
    assert records[0]["logger"] == "atu2.data" and records[1]["level"] == "WARNING"
    assert "boom" in records[2]["exc"]


def test_secret_env_values_and_key_patterns_are_redacted(monkeypatch):
    secret = "s3cr3t-" + "x" * 20  # pragma: allowlist secret
    monkeypatch.setenv("ALPACA_API_SECRET", secret)
    key_id = _fake_key_id()
    buf = io.StringIO()
    configure_logging("run-1", stream=buf, level=logging.DEBUG)
    get_logger("t").info(f"using {key_id} and {secret}", extra={"fields": {"header": secret}})
    out = buf.getvalue()
    assert secret not in out and key_id not in out
    assert out.count(REDACTED) >= 3


def test_redact_assignment_patterns():
    text = "api_key=" + "a1b2c3d4" * 3
    assert REDACTED in redact(text)
    assert redact("nothing secret here") == "nothing secret here"


def test_reconfigure_replaces_handler():
    a, b = io.StringIO(), io.StringIO()
    configure_logging("first", stream=a)
    configure_logging("second", stream=b)
    get_logger("x").info("hello")
    assert a.getvalue() == ""
    assert json.loads(b.getvalue())["run_id"] == "second"
