"""Named error classes. Every invalid input raises one of these; nothing fails silently."""


class ATUError(Exception):
    """Base class of every ATU 2 error."""


# --- configuration --------------------------------------------------------------------
class ConfigError(ATUError):
    """A config file or mapping is invalid."""


class UnknownConfigKeyError(ConfigError):
    """The config contains a key the schema does not define (unknown keys are errors)."""


class MissingConfigKeyError(ConfigError):
    """A required config key is absent."""


class ConfigTypeError(ConfigError):
    """A config value has the wrong type or is not an allowed literal."""


# --- hashing / reproducibility ----------------------------------------------------------
class HashMismatchError(ATUError):
    """A recorded content hash does not match the content (e.g. a frozen file was edited)."""


# --- calendar ---------------------------------------------------------------------------
class CalendarError(ATUError):
    """A date is outside the range the exchange calendar supports."""


# --- data (subclassed per failure type in milestone M1) ----------------------------------
class DataError(ATUError):
    """Base class of data download and validation errors."""


# --- evaluation protocol ---------------------------------------------------------------
class HoldoutAccessError(ATUError):
    """Holdout dates were requested before the final-evaluation flag was set."""


# --- secrets ----------------------------------------------------------------------------
class MissingSecretError(ATUError):
    """A required key is not set in the environment. Keys are never read from files."""
