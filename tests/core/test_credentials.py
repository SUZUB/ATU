"""Alpaca credentials: primary names, aliases, conflicts, and redaction of all four names."""

import io

import pytest

from atu2.core.credentials import CREDENTIAL_ENV_VARS, alpaca_credentials
from atu2.core.errors import ConflictingCredentialError, MissingSecretError
from atu2.core.logging import REDACTED, configure_logging, get_logger

KEY = "key-id-" + "1" * 8  # pragma: allowlist secret
SECRET = "secret-" + "2" * 12  # pragma: allowlist secret


def test_names_are_the_four_agreed_ones():
    assert set(CREDENTIAL_ENV_VARS) == {"APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"}


@pytest.mark.parametrize(
    "env",
    [
        {"APCA_API_KEY_ID": KEY, "APCA_API_SECRET_KEY": SECRET},
        {"ALPACA_API_KEY": KEY, "ALPACA_API_SECRET": SECRET},
        {"APCA_API_KEY_ID": KEY, "ALPACA_API_SECRET": SECRET},
        {"APCA_API_KEY_ID": KEY, "ALPACA_API_KEY": KEY, "APCA_API_SECRET_KEY": SECRET, "ALPACA_API_SECRET": SECRET},
        {"APCA_API_KEY_ID": f"  {KEY}\n", "APCA_API_SECRET_KEY": SECRET},
    ],
)
def test_primary_alias_or_equal_both(env):
    c = alpaca_credentials(env)
    assert (c.key_id, c.secret) == (KEY, SECRET)
    assert KEY not in repr(c) and SECRET not in repr(c)


@pytest.mark.parametrize(
    "env",
    [
        {"APCA_API_KEY_ID": KEY, "ALPACA_API_KEY": KEY + "x", "APCA_API_SECRET_KEY": SECRET},
        {"APCA_API_KEY_ID": KEY, "APCA_API_SECRET_KEY": SECRET, "ALPACA_API_SECRET": SECRET + "x"},
    ],
)
def test_differing_values_raise(env):
    with pytest.raises(ConflictingCredentialError) as e:
        alpaca_credentials(env)
    assert KEY not in str(e.value) and SECRET not in str(e.value)


@pytest.mark.parametrize("env", [{}, {"APCA_API_KEY_ID": KEY}, {"ALPACA_API_SECRET": SECRET}, {"APCA_API_KEY_ID": " "}])
def test_missing_raises(env):
    with pytest.raises(MissingSecretError):
        alpaca_credentials(env)


def test_reads_os_environ_by_default(monkeypatch):
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", KEY)
    monkeypatch.setenv("ALPACA_API_SECRET", SECRET)
    assert alpaca_credentials().key_id == KEY


@pytest.mark.parametrize("name", ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"])
def test_each_name_is_redacted_even_when_short(monkeypatch, name):
    short = "abc12"  # shorter than the heuristic minimum: redacted only because of the explicit name
    monkeypatch.setenv(name, short)
    buf = io.StringIO()
    configure_logging("run", stream=buf)
    get_logger("t").info(f"value={short}", extra={"fields": {"v": short}})
    out = buf.getvalue()
    assert short not in out
    assert out.count(REDACTED) == 2
