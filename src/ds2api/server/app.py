"""Application factory and lifespan wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from ..core.chat import Chat
from ..core.endpoints import Endpoints
from ..core.hif import HifManager
from ..core.http import HTTP_VERSION_1_1, UpstreamHttp
from ..core.pool import AccountPool
from ..core.pow import PowSolver, download_wasm
from ..settings import Config, data_dir
from . import admin, handlers
from .auth import LoginLimiter
from .errors import ApiError
from .middleware import (
    BodyLimitMiddleware,
    RequestIdMiddleware,
    admin_auth_middleware,
    api_key_middleware,
)
from .stats import Stats

logger = logging.getLogger("ds2api.app")


@dataclass
class Ctx:
    config: Config
    config_path: Path
    chat: Chat | None = None
    pool: AccountPool | None = None
    http: UpstreamHttp | None = None
    stats: Stats | None = None
    limiter: LoginLimiter | None = None
    runtime_logs: list | None = None


def _load_pow(config: Config) -> PowSolver | None:
    cache = data_dir() / "cache" / "sha3_wasm_bg.wasm"
    try:
        solver = PowSolver.from_url(config.ds_core.wasm_url, cache)
        logger.info("pow solver ready (wasm cached at %s)", cache)
        return solver
    except Exception as exc:  # noqa: BLE001
        logger.error("pow solver unavailable: %s", exc)
        return None


def create_app(config: Config, config_path: Path | str = "config.toml"):
    ctx = Ctx(config=config, config_path=Path(config_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = UpstreamHttp(
            proxy=config.proxy.url,
            impersonate=config.ds_core.emulation,
            http_version=HTTP_VERSION_1_1 if config.ds_core.api_base.startswith("http://") else None,
        )
        endpoints = Endpoints(http, config.ds_core.api_base)
        pow_solver = _load_pow(config)
        hif = HifManager(http, enabled=config.ds_core.hif_enabled)
        pool = AccountPool(config, endpoints)
        await pool.start()

        ctx.http = http
        ctx.pool = pool
        ctx.chat = Chat(config, pool, endpoints, pow_solver, hif)
        ctx.stats = Stats.load(data_dir() / "stats.json")
        ctx.limiter = LoginLimiter()
        ctx.runtime_logs = []

        logger.info("listening on %s:%s", config.server.host, config.server.port)
        try:
            yield
        finally:
            if ctx.chat is not None:
                await ctx.chat.shutdown()
            await pool.shutdown()
            await http.close()
            if ctx.stats is not None:
                try:
                    ctx.stats.persist_now()
                except OSError as exc:
                    logger.warning("failed to persist stats: %s", exc)

    app = FastAPI(title="DS2Api", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ctx = ctx

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        request_id = (request.scope.get("state") or {}).get("request_id")
        return exc.response(request_id=request_id)

    @app.get("/health")
    async def health():
        return await handlers.health()

    @app.get("/")
    async def root():
        return RedirectResponse("/admin")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handlers.chat_completions(request, ctx)

    @app.get("/v1/models")
    async def list_models(request: Request):
        return await handlers.list_models(request, ctx)

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str):
        return await handlers.get_model(model_id, ctx)

    # -- admin ---------------------------------------------------------
    @app.post("/admin/api/setup")
    async def admin_setup(request: Request):
        return await admin.setup(request, ctx)

    @app.post("/admin/api/login")
    async def admin_login(request: Request):
        return await admin.login(request, ctx)

    @app.get("/admin/api/status")
    async def admin_status():
        return await admin.status(ctx)

    @app.get("/admin/api/account-statuses-detailed")
    async def admin_accounts():
        return await admin.account_statuses(ctx)

    @app.get("/admin/api/stats")
    async def admin_stats():
        return await admin.stats(ctx)

    @app.get("/admin/api/models")
    async def admin_models():
        return await admin.models(ctx)

    @app.get("/admin/api/config")
    async def admin_get_config():
        return await admin.get_config(ctx)

    @app.put("/admin/api/config")
    async def admin_put_config(request: Request):
        return await admin.put_config(request, ctx)

    @app.get("/admin/api/logs")
    async def admin_logs(request: Request):
        return await admin.logs(request, ctx)

    @app.get("/admin/api/runtime-logs")
    async def admin_runtime_logs(request: Request):
        return await admin.runtime_logs(request, ctx)

    @app.get("/admin/api/sessions")
    async def admin_sessions():
        return await admin.sessions(ctx)

    origins = config.server.cors_origins or ["*"]
    cors = CORSMiddleware(
        app,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-ds-account"],
    )

    asgi = cors
    asgi = api_key_middleware(asgi, lambda: ctx.config)
    asgi = admin_auth_middleware(asgi, lambda: ctx.config)
    asgi = BodyLimitMiddleware(asgi)
    asgi = RequestIdMiddleware(asgi)
    return asgi
