"""
Shared FastAPI dependencies.

`get_current_user` is the gate on every protected endpoint. Writing it once and
declaring it as a parameter means authentication can never be forgotten on a new
route -- if the dependency isn't there, the route simply has no user to work
with, which is immediately obvious.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Paper, User
from app.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise unauthorized

    user = db.get(User, user_id)
    if user is None:
        raise unauthorized

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_owned_paper(paper_id: str, db: DbSession, user: CurrentUser) -> Paper:
    """Fetch a paper, but only if it belongs to the caller.

    This is the fix for IDOR (Insecure Direct Object Reference): without the
    ownership check, anyone could read another user's paper just by guessing its
    id. Returning 404 rather than 403 also avoids confirming that the id exists.
    """
    paper = db.get(Paper, paper_id)
    if paper is None or paper.user_id != user.id:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


OwnedPaper = Annotated[Paper, Depends(get_owned_paper)]
