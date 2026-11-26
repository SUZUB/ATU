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


# --- interface contracts -----------------------------------------------------------------
class ContractError(ATUError):
    """An object passed between layers breaks its interface contract."""


# --- data (M1) ----------------------------------------------------------------------------
class DataError(ATUError):
    """Base class of data download and validation errors."""


class DataSourceError(DataError):
    """A data provider request failed (HTTP error, bad payload, retries exhausted)."""


class RateLimitError(DataSourceError):
    """The provider kept answering HTTP 429 after every retry."""


class CacheRewriteError(DataError):
    """Cached history and a fresh re-download disagree in a way that cannot be repaired."""


class CrossCheckError(DataError):
    """The second price source disagrees with the primary data beyond the tolerance."""


class DataValidationError(DataError):
    """Base class of the named validation errors (FR-02); one subclass per failure type."""


class MissingColumnError(DataValidationError):
    """A required column is absent."""


class MissingValueError(DataValidationError):
    """A price or volume is NaN or infinite."""


class NonPositivePriceError(DataValidationError):
    """A price is zero or negative."""


class HighBelowLowError(DataValidationError):
    """A bar's high is below its low."""


class PriceOutsideRangeError(DataValidationError):
    """A bar's open or close lies outside its own high-low range."""


class NegativeVolumeError(DataValidationError):
    """Volume is negative."""


class ZeroVolumeRunError(DataValidationError):
    """Volume is zero for more consecutive days than allowed."""


class DuplicateDateError(DataValidationError):
    """The same date appears twice."""


class UnsortedDatesError(DataValidationError):
    """Dates are not in increasing order."""


class NonTradingDayError(DataValidationError):
    """A bar is dated on a weekend or NYSE holiday."""


class CalendarGapError(DataValidationError):
    """More consecutive NYSE trading days are missing than allowed."""


class OutlierMoveError(DataValidationError):
    """A daily adjusted move exceeds the threshold and has not been approved by a human."""


class UnadjustedDataError(DataValidationError):
    """The data is not split- and dividend-adjusted."""


# --- evaluation protocol ---------------------------------------------------------------
class HoldoutAccessError(ATUError):
    """Holdout dates were requested before the final-evaluation flag was set."""


# --- secrets ----------------------------------------------------------------------------
class MissingSecretError(ATUError):
    """A required key is not set in the environment. Keys are never read from files."""


class ConflictingCredentialError(ATUError):
    """A primary key variable and its alias are both set but hold different values."""
