"""
Password hashing and JWT tokens.

Two things worth knowing if you're asked about this file:

1. Passwords are hashed with bcrypt, never encrypted and never stored plain.
   bcrypt is intentionally slow and salts every hash automatically, so two users
   with the same password get different stored values.

2. A JWT is three base64 parts: header.payload.signature. The payload is *not*
   secret -- anyone can read it. What the signature guarantees is that nobody
   tampered with it, because only the server knows SECRET_KEY.
"""

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings


def hash_password(plain: str) -> str:
    # bcrypt operates on bytes and ignores anything past 72 bytes, so we
    # reject over-long passwords rather than silently truncating them.
    raw = plain.encode("utf-8")
    if len(raw) > 72:
        raise ValueError("Password must be at most 72 bytes long.")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in the DB -- treat as a failed login, don't crash.
        return False


def create_access_token(user_id: str) -> str:
    """Issue a signed token identifying the user.

    'sub' (subject) is the standard claim for "who this token is about".
    'exp' is checked automatically by PyJWT on decode.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: str, version: int) -> str:
    """A long-lived token that can only be exchanged for an access token.

    `typ` distinguishes it from an access token so one cannot be passed off as
    the other: a refresh token presented as a Bearer header is rejected by
    `decode_access_token`, and vice versa.

    `ver` is the user's token version. Bumping it on the user row invalidates
    every refresh token issued before -- that is how "log out everywhere" and
    password changes revoke sessions without a server-side token table.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "typ": "refresh",
        "ver": version,
        # Unique per token, so two issued in the same second still differ and
        # a rotated token is never byte-identical to the one it replaced.
        "jti": secrets.token_hex(8),
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_refresh_token(token: str) -> tuple[str, int] | None:
    """(user id, token version) from a valid refresh token, else None."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "refresh":
        return None
    sub, ver = payload.get("sub"), payload.get("ver")
    if not isinstance(sub, str) or not isinstance(ver, int):
        return None
    return sub, ver


def decode_access_token(token: str) -> str | None:
    """Return the user id inside a valid token, or None if it's bad/expired."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        if payload.get("typ") == "refresh":
            return None  # a refresh token is not a credential for API calls
    except jwt.PyJWTError:
        return None
    return payload.get("sub")
