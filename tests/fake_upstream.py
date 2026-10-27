"""Fake DeepSeek upstream for tests.

Runs a real uvicorn server on an ephemeral port so streaming, cancellation and
connection teardown behave exactly like production.

It records every request so tests can assert on headers and bodies, and the SSE
frames are scriptable.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

REAL_CHALLENGE = {
    "algorithm": "DeepSeekHashV1",
    "challenge": "7ffc9d19b6eed96a6fca68f8ffe30ee61035d4959e4180f187bf85b356016a96",
    "salt": "3bde54628ea8413fee87",
    "signature": "ce4678cf7a1290c2a7ac88c4195a5b8497e5fc4b0e8044e804f5a6f3af6fe462",
    "difficulty": 144000,
    "expire_at": 1775380966945,
    "target_path": "/api/v0/chat/completion",
}


def envelope(biz_data: Any, *, biz_code: int = 0, biz_msg: str = "") -> dict:
    return {
        "code": 0,
        "msg": "",
        "data": {"biz_code": biz_code, "biz_msg": biz_msg, "biz_data": biz_data},
    }


@dataclass
class FakeState:
    requests: list[dict] = field(default_factory=list)
    think_text: str = ""
    response_text: str = "hello"
    usage: int = 5
    status: str = "FINISHED"
    extra_frames: list[str] = field(default_factory=list)
    fail_login_biz_code: int | None = None
    rate_limit: bool = False
    session_seq: int = 0
    uploaded: list[str] = field(default_factory=list)

    def record(self, path: str, *, headers: dict, body: Any = None) -> None:
        self.requests.append({"path": path, "headers": headers, "body": body})

    def by_path(self, path: str) -> list[dict]:
        return [r for r in self.requests if r["path"] == path]

    def last(self, path: str) -> dict | None:
        items = self.by_path(path)
        return items[-1] if items else None


def _data(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _event(name: str, payload: dict) -> str:
    return f"event: {name}\n" + _data(payload)


def build_frames(state: FakeState) -> list[str]:
    frames = [
        _event(
            "ready",
            {"request_message_id": 1, "response_message_id": 2, "model_type": "default"},
        )
    ]
    if state.rate_limit:
        frames.append(
            _event(
                "hint",
                {
                    "type": "error",
                    "content": "rate limit reached",
                    "finish_reason": "rate_limit_reached",
                },
            )
        )
        return frames

    if state.think_text:
        frames.append(
            _data(
                {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "THINK", "content": state.think_text}],
                }
            )
        )
    if state.response_text:
        frames.append(
            _data(
                {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "RESPONSE", "content": state.response_text}],
                }
            )
        )
    frames.extend(state.extra_frames)
    frames.append(
        _data(
            {
                "p": "response",
                "o": "BATCH",
                "v": [
                    {"p": "accumulated_token_usage", "v": state.usage},
                    {"p": "quasi_status", "v": state.status},
                ],
            }
        )
    )
    frames.append(
        _data({"p": "response/status", "o": "SET", "v": state.status})
    )
    frames.append(_event("update_session", {"updated_at": 1775386361.5}))
    frames.append(_event("close", {"click_behavior": "none", "auto_resume": False}))
    return frames


def make_app(state: FakeState) -> Starlette:
    async def headers_of(request: Request) -> dict:
        return {k.lower(): v for k, v in request.headers.items()}

    async def login(request: Request) -> JSONResponse:
        body = await request.json()
        state.record("/users/login", headers=await headers_of(request), body=body)
        if state.fail_login_biz_code is not None:
            return JSONResponse(envelope(None, biz_code=state.fail_login_biz_code, biz_msg="nope"))
        return JSONResponse(
            envelope({"user": {"id": "u1", "token": "tok-1", "email": body.get("email")}})
        )

    async def check_device(request: Request) -> JSONResponse:
        state.record("/users/auth_token/check_device", headers=await headers_of(request),
                     body=await request.json())
        return JSONResponse(envelope({"rotate": None}))

    async def session_create(request: Request) -> JSONResponse:
        state.record("/chat_session/create", headers=await headers_of(request))
        state.session_seq += 1
        return JSONResponse(envelope({"chat_session": {"id": f"sess-{state.session_seq}"}}))

    async def session_delete(request: Request) -> JSONResponse:
        state.record("/chat_session/delete", headers=await headers_of(request),
                     body=await request.json())
        return JSONResponse(envelope(None))

    async def pow_challenge(request: Request) -> JSONResponse:
        body = await request.json()
        state.record("/chat/create_pow_challenge", headers=await headers_of(request), body=body)
        challenge = dict(REAL_CHALLENGE, target_path=body.get("target_path", ""))
        return JSONResponse(envelope({"challenge": challenge}))

    async def completion(request: Request) -> Any:
        body = await request.json()
        state.record("/chat/completion", headers=await headers_of(request), body=body)
        frames = build_frames(state)

        async def gen():
            for frame in frames:
                yield frame.encode()
                await asyncio.sleep(0.005)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def stop_stream(request: Request) -> JSONResponse:
        state.record("/chat/stop_stream", headers=await headers_of(request),
                     body=await request.json())
        return JSONResponse(envelope(None))

    async def upload(request: Request) -> JSONResponse:
        form = await request.form()
        name = ""
        for key in form:
            value = form[key]
            name = getattr(value, "filename", "") or ""
        state.record("/file/upload_file", headers=await headers_of(request), body=name)
        state.uploaded.append(name)
        return JSONResponse(envelope({"id": f"file-{len(state.uploaded)}", "status": "PENDING",
                                      "file_name": name, "file_size": 3}))

    async def fetch_files(request: Request) -> JSONResponse:
        state.record("/file/fetch_files", headers=await headers_of(request),
                     body=dict(request.query_params))
        return JSONResponse(
            envelope({"files": [{"id": "file-1", "status": "SUCCESS", "file_name": "x.txt",
                                 "file_size": 3}]})
        )

    app = Starlette(
        routes=[
            Route("/api/v0/users/login", login, methods=["POST"]),
            Route("/api/v0/users/auth_token/check_device", check_device, methods=["POST"]),
            Route("/api/v0/chat_session/create", session_create, methods=["POST"]),
            Route("/api/v0/chat_session/delete", session_delete, methods=["POST"]),
            Route("/api/v0/chat/create_pow_challenge", pow_challenge, methods=["POST"]),
            Route("/api/v0/chat/completion", completion, methods=["POST"]),
            Route("/api/v0/chat/stop_stream", stop_stream, methods=["POST"]),
            Route("/api/v0/file/upload_file", upload, methods=["POST"]),
            Route("/api/v0/file/fetch_files", fetch_files, methods=["GET"]),
        ]
    )
    return app


async def start_server(app, host: str = "127.0.0.1") -> tuple[Any, int]:
    """Run ``app`` on an ephemeral port; returns ``(server, port)``."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    if not server.started:
        task.cancel()
        raise RuntimeError("server failed to start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, port
