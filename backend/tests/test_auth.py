"""
Tests for the auth flow.

Each test gets a fresh in-memory database, so tests never interfere with each
other and never touch the real paperlens.db file.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",  # in-memory
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # keeps one connection so the DB survives between calls
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_root_signposts_instead_of_404ing(client):
    from app.main import STATIC_DIR

    if STATIC_DIR.is_dir():
        pytest.skip("a built frontend is present, so / serves index.html instead")
    """Platform health probes hit / -- it should answer, not look broken."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["health"] == "/api/health"


def test_register_returns_token_and_user(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "Ada@Example.com", "password": "supersecret1", "full_name": "Ada"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    # Emails are normalised to lowercase on the way in.
    assert body["user"]["email"] == "ada@example.com"


def test_duplicate_email_rejected(client):
    payload = {"email": "ada@example.com", "password": "supersecret1"}
    client.post("/api/auth/register", json=payload)
    res = client.post("/api/auth/register", json=payload)
    assert res.status_code == 409


def test_short_password_rejected(client):
    res = client.post(
        "/api/auth/register", json={"email": "a@b.com", "password": "short"}
    )
    assert res.status_code == 422  # Pydantic validation, before any DB work


def test_login_success_and_failure(client):
    client.post(
        "/api/auth/register",
        json={"email": "ada@example.com", "password": "supersecret1"},
    )

    ok = client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "supersecret1"},
    )
    assert ok.status_code == 200

    bad = client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "wrongpassword"},
    )
    assert bad.status_code == 401


def test_me_requires_valid_token(client):
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code
        == 401
    )

    token = client.post(
        "/api/auth/register",
        json={"email": "ada@example.com", "password": "supersecret1"},
    ).json()["access_token"]

    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["email"] == "ada@example.com"


# ---------- refresh tokens ----------

def test_login_sets_httponly_refresh_cookie_and_refresh_rotates_it(client):
    client.post("/api/auth/register", json={"email": "r@example.com", "password": "supersecret1"})
    response = client.post("/api/auth/login", json={"email": "r@example.com", "password": "supersecret1"})
    assert response.status_code == 200
    cookie = response.headers.get("set-cookie", "")
    assert "paperlens_refresh=" in cookie
    assert "HttpOnly" in cookie and "Path=/api/auth" in cookie and "SameSite=lax" in cookie.replace("SameSite=Lax", "SameSite=lax")

    first = client.cookies.get("paperlens_refresh")
    refreshed = client.post("/api/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["user"]["email"] == "r@example.com"
    assert client.cookies.get("paperlens_refresh") != first  # rotated

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"})
    assert me.status_code == 200


def test_refresh_token_cannot_be_used_as_an_access_token(client):
    client.post("/api/auth/register", json={"email": "r2@example.com", "password": "supersecret1"})
    refresh_cookie = client.cookies.get("paperlens_refresh")
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {refresh_cookie}"})
    assert me.status_code == 401


def test_logout_everywhere_revokes_older_refresh_tokens(client):
    client.post("/api/auth/register", json={"email": "r3@example.com", "password": "supersecret1"})
    old_cookie = client.cookies.get("paperlens_refresh")

    assert client.post("/api/auth/logout", params={"everywhere": "true"}).status_code == 204

    client.cookies.set("paperlens_refresh", old_cookie)
    assert client.post("/api/auth/refresh").status_code == 401


def test_refresh_without_cookie_is_401(client):
    client.cookies.clear()
    assert client.post("/api/auth/refresh").status_code == 401


def test_security_headers_are_present(client):
    response = client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
