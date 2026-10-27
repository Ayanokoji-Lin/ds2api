"""Pure-ASGI middleware.

``BaseHTTPMiddleware`` is deliberately avoided:

* it adds a memory-channel round trip per chunk, hurting SSE latency/backpressure
* it swallows ``CancelledError``, so the ``finally`` that stops the upstream
  session never runs -> leaked sessions
* it cannot reliably rewrite the status/headers before a streaming response starts
"""

from __future__ import annotations

import itertools
import logging
from typing import Awaitable, Callable

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from ..settings import Config
from . import auth
from .errors import unauthorized

logger = logging.getLogger("ds2api.middleware")

MAX_BODY = 10_000_000

_request_counter = itertools.count(1)

Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


class RequestIdMiddleware:
    """Generate ``req-{hex}`` and echo it back on every response, including errors."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = f"req-{next(_request_counter):x}"
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("x-request-id", request_id)
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodyLimitMiddleware:
    """Reject oversized bodies before they are read."""

    def __init__(self, app, limit: int = MAX_BODY) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"content-length")
            if raw:
                try:
                    length = int(raw)
                except ValueError:
                    length = 0
                if length > self.limit:
                    response = PlainTextResponse("request body too large", status_code=413)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def api_key_middleware(app, get_config: Callable[[], Config]):
    """Protect ``/v1/*`` with the configured API keys."""

    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith("/v1/"):
            await app(scope, receive, send)
            return

        request = Request(scope, receive)
        token = auth.extract_api_token(request)
        if not auth.is_authorized(get_config(), token):
            error = unauthorized()
            request_id = (scope.get("state") or {}).get("request_id")
            await error.response(request_id=request_id)(scope, receive, send)
            return
        await app(scope, receive, send)

    return middleware


def admin_auth_middleware(app, get_config: Callable[[], Config]):
    """Protect ``/admin/api/*`` except setup/login with a JWT."""

    open_paths = {
        "/admin/api/setup",
        "/admin/api/login",
        "/admin/api/setup-status",
    }

    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith("/admin/api/") or path in open_paths:
            await app(scope, receive, send)
            return

        request = Request(scope, receive)
        token = auth.extract_bearer(request)
        config = get_config()
        ok = bool(token) and auth.verify_token(
            config.admin.jwt_secret, token or "", config.admin.jwt_issued_at
        )
        if not ok:
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await app(scope, receive, send)

    return middleware
