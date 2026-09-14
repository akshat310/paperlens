"""
Registration, login, refresh, logout, and 'who am I'.

Session model
-------------
Two tokens, with different jobs:

* The **access token** is short-lived (minutes) and travels in the
  Authorization header. The browser keeps it in memory only -- never in
  localStorage, where any script that runs on the page can read it.
* The **refresh token** is long-lived (days) and travels in an httpOnly,
  SameSite cookie scoped to /api/auth. JavaScript cannot read it, so an
  injected script cannot steal the session; and because it is only ever sent
  to /api/auth, a CSRF attempt against another endpoint gains nothing.

On page load the SPA calls /auth/refresh: the cookie goes along automatically,
and a fresh access token comes back. Refresh tokens are rotated on every use
and carry the user's token version, so logging out (or a password change) can
invalidate all of them by bumping one integer on the user row.
"""

from fastapi import APIRouter, Cookie, HTTPException, Response, status
from sqlalchemy import select

from app import ratelimit
from app.config import settings
from app.deps import CurrentUser, DbSession
from app.models import User
from app.schemas import Token, UserCreate, UserLogin, UserOut
from app.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "paperlens_refresh"


def _issue(response: Response, user: User) -> Token:
    """Set the refresh cookie and return the access token payload."""
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=create_refresh_token(user.id, user.token_version),
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,
        # Lax: sent on same-site requests and top-level navigations, which is
        # all a same-origin SPA needs; blocks cross-site POSTs.
        samesite="lax",
        # Secure requires HTTPS. Render terminates TLS; local dev is http.
        secure=not settings.DEBUG,
        path="/api/auth",
    )
    return Token(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


def _clear(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path="/api/auth")


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: DbSession, response: Response) -> Token:
    existing = db.scalar(select(User).where(User.email == payload.email.lower()))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email is already registered.")

    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return _issue(response, user)


@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: DbSession, response: Response) -> Token:
    # Per email, not per IP: the thing being protected is one account's
    # password, and a proxy shares an IP across many honest users.
    ratelimit.check(ratelimit.LOGIN, payload.email.lower(), "login attempts")
    user = db.scalar(select(User).where(User.email == payload.email.lower()))

    # Deliberately identical error for "no such user" and "wrong password".
    # Distinguishing them would let an attacker enumerate valid email addresses.
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    return _issue(response, user)


@router.post("/refresh", response_model=Token)
def refresh(
    db: DbSession,
    response: Response,
    paperlens_refresh: str | None = Cookie(default=None),
) -> Token:
    """Exchange the refresh cookie for a new access token (and a rotated cookie).

    401 on a missing, expired, mistyped or revoked token. The SPA treats that
    as "not logged in" -- it is the normal state on a first visit.
    """
    decoded = decode_refresh_token(paperlens_refresh) if paperlens_refresh else None
    if decoded is None:
        _clear(response)
        raise HTTPException(status_code=401, detail="Not authenticated.")

    user_id, version = decoded
    user = db.get(User, user_id)
    if user is None or user.token_version != version:
        _clear(response)
        raise HTTPException(status_code=401, detail="Session expired.")

    return _issue(response, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    db: DbSession,
    response: Response,
    everywhere: bool = False,
    paperlens_refresh: str | None = Cookie(default=None),
) -> None:
    """Clear the cookie; with ?everywhere=true also revoke every other session.

    Deliberately does not require a valid access token: a user whose access
    token has expired must still be able to log out.
    """
    if everywhere and paperlens_refresh:
        decoded = decode_refresh_token(paperlens_refresh)
        if decoded is not None:
            user = db.get(User, decoded[0])
            if user is not None:
                user.token_version += 1
                db.commit()
    _clear(response)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
