from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.guardrails import APIKeyAuthMiddleware, FixedWindowRateLimitMiddleware


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat")
    def chat() -> dict:
        return {"ok": True}

    return app


def test_rate_limit_blocks_after_configured_window_limit():
    app = _app()
    app.add_middleware(
        FixedWindowRateLimitMiddleware,
        enabled=True,
        requests=2,
        window_seconds=60,
        protected_path_prefixes=("/chat",),
    )
    client = TestClient(app)

    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 200

    resp = client.post("/chat")

    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limit_exceeded"
    assert 1 <= int(resp.headers["retry-after"]) <= 60


def test_rate_limit_ignores_unprotected_paths():
    app = _app()
    app.add_middleware(
        FixedWindowRateLimitMiddleware,
        enabled=True,
        requests=0,
        window_seconds=60,
        protected_path_prefixes=("/chat",),
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 200


def test_api_key_auth_blocks_missing_key_when_enabled():
    app = _app()
    app.add_middleware(
        APIKeyAuthMiddleware,
        enabled=True,
        api_keys=("secret",),
        exempt_path_prefixes=("/health",),
    )
    client = TestClient(app)

    resp = client.post("/chat")

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "auth_required"


def test_api_key_auth_accepts_valid_key_and_exempts_health():
    app = _app()
    app.add_middleware(
        APIKeyAuthMiddleware,
        enabled=True,
        api_keys=("secret",),
        exempt_path_prefixes=("/health",),
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/chat", headers={"x-api-key": "secret"}).status_code == 200
