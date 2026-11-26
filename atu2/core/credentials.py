"""Alpaca API credentials from environment variables only (locked decision D8, NFR-04).

Accepted names:
    primary  APCA_API_KEY_ID      / APCA_API_SECRET_KEY   (Alpaca's own names, used by ORB v2)
    aliases  ALPACA_API_KEY       / ALPACA_API_SECRET     (the names in docs/BUILD_PLAN.md)

For each of the two values: if the primary and the alias are both set and differ,
ConflictingCredentialError is raised; nothing is picked silently. If neither is set,
MissingSecretError is raised. Values are never read from files and never shown in a repr.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from atu2.core.errors import ConflictingCredentialError, MissingSecretError

KEY_ID_VARS: tuple[str, str] = ("APCA_API_KEY_ID", "ALPACA_API_KEY")
SECRET_VARS: tuple[str, str] = ("APCA_API_SECRET_KEY", "ALPACA_API_SECRET")
CREDENTIAL_ENV_VARS: tuple[str, ...] = (*KEY_ID_VARS, *SECRET_VARS)


@dataclass(frozen=True)
class AlpacaCredentials:
    """Key id and secret; both are hidden from repr() so they cannot leak into logs."""

    key_id: str = field(repr=False)
    secret: str = field(repr=False)


def _resolve(names: tuple[str, str], env: Mapping[str, str]) -> str:
    primary, alias = names
    a = env.get(primary, "").strip()
    b = env.get(alias, "").strip()
    if a and b and a != b:
        raise ConflictingCredentialError(
            f"{primary} and {alias} are both set to different values; unset one of them or make them equal"
        )
    value = a or b
    if not value:
        raise MissingSecretError(
            f"{primary} (or its alias {alias}) is not set. Set it in your shell or Colab session; "
            "never put keys in files, notebooks or chat."
        )
    return value


def alpaca_credentials(env: Mapping[str, str] | None = None) -> AlpacaCredentials:
    """Reads the Alpaca key id and secret from `env` (default: os.environ)."""
    source = os.environ if env is None else env
    return AlpacaCredentials(key_id=_resolve(KEY_ID_VARS, source), secret=_resolve(SECRET_VARS, source))
