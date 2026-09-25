"""
Environment-variable readers that tolerate blank or malformed values.

Hosting dashboards make it easy to create a variable with an empty value (for
example by pasting .env.example). os.getenv() then returns "" instead of the
default, and float("") crashed the app on import.
"""
import logging
import os

logger = logging.getLogger(__name__)


def env_str(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring %s=%r (not a number); using %s", name, value, default)
        return default


def env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring %s=%r (not an integer); using %s", name, value, default)
        return default
