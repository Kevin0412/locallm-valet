"""Password hashing & API-key generation (pure stdlib, no new deps).

Passwords are stored as PBKDF2-SHA256 hashes in the format::

    pbkdf2:sha256:<iterations>$<base64url-salt>$<base64url-hash>

Plaintext passwords already present in a config file keep working
(backwards compatibility) — ``verify_password`` falls back to a direct
comparison for anything that is not a hash; the settings flow upgrades it
to a hash on first save. Plaintext is never written back by this module.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

# OWASP 2023 recommendation for PBKDF2-SHA256.
DEFAULT_ITERATIONS = 260_000

_PREFIX = "pbkdf2:"


def is_hashed(value: str) -> bool:
    """True when ``value`` looks like a stored PBKDF2 hash."""
    return isinstance(value, str) and value.startswith(_PREFIX)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Hash ``password`` with a random 16-byte salt (PBKDF2-SHA256)."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_PREFIX}sha256:{iterations}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    """Check ``password`` against ``stored``.

    Hashed values go through PBKDF2 (constant-time digest comparison);
    anything else is compared directly as plaintext so pre-existing config
    files keep authenticating until the next password change.
    """
    if not stored or not isinstance(stored, str):
        return False
    if not is_hashed(stored):
        return hmac.compare_digest(password.encode("utf-8"), stored.encode("utf-8"))

    try:
        algo_tag, rest = stored[len(_PREFIX):].split("$", 1)
        algo, iter_text = algo_tag.split(":")
        salt_b64, hash_b64 = rest.split("$", 1)
        if algo != "sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _b64decode(salt_b64), int(iter_text)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _b64decode(hash_b64))


def generate_api_key() -> str:
    """A fresh Bearer key: ``sk-`` + 32 hex chars (128 bits of entropy)."""
    return "sk-" + secrets.token_hex(32)
