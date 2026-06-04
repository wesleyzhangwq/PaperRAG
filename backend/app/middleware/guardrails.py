"""Request guardrails for public API endpoints."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def _error_payload(code: str, user_message: str, action_hint: str, retryable: bool) -> dict:
    return {
        "error": {
            "code": code,
            "user_message": user_message,
            "action_hint": action_hint,
            "retryable": retryable,
        }
    }


def _path_matches(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


class FixedWindowRateLimitMiddleware(BaseHTTPMiddleware):
    """Small in-process fixed-window limiter for chat/upload protection.

    This is intentionally process-local. It caps obvious abuse in single-node
    deployments and can later be swapped for Redis without changing routers.
    """

    def __init__(
        self,
        app,
        *,
        enabled: bool,
        requests: int,
        window_seconds: int,
        protected_path_prefixes: tuple[str, ...],
        key_func: Callable[[Request], str] | None = None,
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.requests = max(0, requests)
        self.window_seconds = max(1, window_seconds)
        self.protected_path_prefixes = protected_path_prefixes
        self.key_func = key_func or self._default_key
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if (
            not self.enabled
            or not _path_matches(request.url.path, self.protected_path_prefixes)
        ):
            return await call_next(request)

        now = time.monotonic()
        key = self.key_func(request)
        with self._lock:
            started_at, count = self._buckets.get(key, (now, 0))
            elapsed = now - started_at
            if elapsed >= self.window_seconds:
                started_at, count = now, 0
            if count >= self.requests:
                retry_after = max(1, int(self.window_seconds - elapsed))
                return JSONResponse(
                    status_code=429,
                    content=_error_payload(
                        "rate_limit_exceeded",
                        "请求过于频繁，请稍后再试。",
                        "稍等一段时间后重试，或降低请求频率。",
                        True,
                    ),
                    headers={"Retry-After": str(retry_after)},
                )
            self._buckets[key] = (started_at, count + 1)

        return await call_next(request)

    @staticmethod
    def _default_key(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Optional API-key protection for non-public routes."""

    def __init__(
        self,
        app,
        *,
        enabled: bool,
        api_keys: tuple[str, ...],
        exempt_path_prefixes: tuple[str, ...],
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.api_keys = {key for key in api_keys if key}
        self.exempt_path_prefixes = exempt_path_prefixes

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if (
            not self.enabled
            or _path_matches(request.url.path, self.exempt_path_prefixes)
        ):
            return await call_next(request)

        provided = request.headers.get("x-api-key", "").strip()
        if not self.api_keys or provided not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content=_error_payload(
                    "auth_required",
                    "需要有效的访问密钥。",
                    "请配置 X-API-Key 请求头后重试。",
                    False,
                ),
            )
        return await call_next(request)
