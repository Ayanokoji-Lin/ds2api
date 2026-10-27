"""Authentication: client API keys and the admin JWT.

API key rules (matching the Rust project):

* ``Authorization: Bearer <key>`` wins, and the ``Bearer `` prefix is
  **case-sensitive** - ``bearer`` is rejected
* ``x-api-key`` is the fallback (what Claude-style clients send)
* there is no anonymous bypass: an empty key list means every /v1 request 401s
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

import jwt
from starlette.requests import Request

from ..settings import Config

JWT_TTL = 24 * 3600
JWT_LEEWAY = 60
BCRYPT_ROUNDS = 12
MAX_LOGIN_FAILURES = 5
LOGIN_LOCK_SECS = 300


def extract_bearer(request: Request) -> str | None:
    raw = request.headers.get("authorization")
    if not raw:
        return None
    if not raw.startswith("Bearer "):
        return None
    token = raw[len("Bearer ") :]
    return token or None


def extract_api_token(request: Request) -> str | None:
    token = extract_bearer(request)
    if token:
        return token
    header = request.headers.get("x-api-key")
    if header:
        return header
    return None


def is_authorized(config: Config, token: str | None) -> bool:
    if not token:
        return False
    return config.is_valid_api_key(token)


# -- admin password --------------------------------------------------------


def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(password: str, hashed: str) -> bool:
    import bcrypt

    if not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


def new_jwt_secret() -> str:
    return secrets.token_hex(32)


# -- JWT -------------------------------------------------------------------


def issue_token(secret: str, issued_at: int) -> str:
    now = int(time.time())
    payload = {"sub": "admin", "iat": now, "exp": now + JWT_TTL}
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_token(secret: str, token: str, min_issued_at: int) -> bool:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            leeway=JWT_LEEWAY,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError:
        return False
    if payload.get("sub") != "admin":
        return False
    issued = payload.get("iat", 0)
    # Tokens issued before the last password change are revoked.
    return int(issued) >= int(min_issued_at)


class LoginLimiter:
    """Global failure counter (not per-IP), matching the Rust implementation."""

    def __init__(self) -> None:
        self._failures = 0
        self._locked_until = 0.0

    def is_locked(self) -> bool:
        return time.monotonic() < self._locked_until

    def remaining(self) -> int:
        return max(0, int(self._locked_until - time.monotonic()))

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= MAX_LOGIN_FAILURES:
            self._locked_until = time.monotonic() + LOGIN_LOCK_SECS
            self._failures = 0

    def reset(self) -> None:
        self._failures = 0
        self._locked_until = 0.0
