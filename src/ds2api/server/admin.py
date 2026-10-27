"""Admin API.

Envelope is a plain ``{"error": "..."}`` string (not the OpenAI shape), and the
status code is what the dashboard branches on, so keep them stable:

    403 password already set / not set
    429 too many attempts
    400 password too short / bad JSON
    401 wrong password
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from ..openai import models as models_api
from ..settings import Config
from . import auth
from .errors import ApiError

logger = logging.getLogger("ds2api.admin")


def _err(status: int, message: str) -> ApiError:
    return ApiError(status, message, "admin_error", "admin_error")


async def _json(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise _err(400, f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _err(400, "body must be an object")
    return payload


def _masked_config(config: Config) -> dict:
    data = config.model_dump(mode="python")
    data["admin"] = {
        "password_set": bool(config.admin.password_hash),
        "jwt_issued_at": config.admin.jwt_issued_at,
    }
    return data


async def setup(request: Request, ctx) -> JSONResponse:
    payload = await _json(request)
    config: Config = ctx.config
    if config.admin.password_hash:
        raise _err(403, "密码已设置")
    password = str(payload.get("password", ""))
    if len(password) < 6:
        raise _err(400, "密码至少 6 位")

    config.admin.password_hash = auth.hash_password(password)
    config.admin.jwt_secret = auth.new_jwt_secret()
    config.admin.jwt_issued_at = int(__import__("time").time())
    config.save(ctx.config_path)
    logger.info("admin password configured")
    return JSONResponse({"token": auth.issue_token(config.admin.jwt_secret, config.admin.jwt_issued_at)})


async def login(request: Request, ctx) -> JSONResponse:
    payload = await _json(request)
    config: Config = ctx.config
    if not config.admin.password_hash:
        raise _err(403, "未设置密码")

    limiter = ctx.limiter
    if limiter.is_locked():
        raise _err(429, f"尝试次数过多，请 {limiter.remaining()} 秒后再试")

    password = str(payload.get("password", ""))
    if not auth.verify_password(password, config.admin.password_hash):
        limiter.record_failure()
        raise _err(401, "密码错误")

    limiter.reset()
    return JSONResponse({"token": auth.issue_token(config.admin.jwt_secret, config.admin.jwt_issued_at)})


async def status(ctx) -> JSONResponse:
    pool = ctx.pool
    counts = pool.counts() if pool else {}
    return JSONResponse(
        {
            "accounts": len(pool.accounts) if pool else 0,
            "idle": counts.get("idle", 0),
            "busy": counts.get("busy", 0),
            "error": counts.get("error", 0),
            "invalid": counts.get("invalid", 0),
        }
    )


async def account_statuses(ctx) -> JSONResponse:
    return JSONResponse({"accounts": ctx.pool.statuses() if ctx.pool else []})


async def stats(ctx) -> JSONResponse:
    return JSONResponse(ctx.stats.snapshot())


async def models(ctx) -> JSONResponse:
    return JSONResponse(models_api.list_models(ctx.config))


async def get_config(ctx) -> JSONResponse:
    return JSONResponse(_masked_config(ctx.config))


async def put_config(request: Request, ctx) -> JSONResponse:
    payload = await _json(request)
    old = ctx.config.model_dump(mode="python")

    # Passwords are never echoed back, so an empty value means "keep existing".
    incoming_core = payload.get("ds_core") or {}
    incoming_accounts = incoming_core.get("accounts") or []
    old_accounts = (old.get("ds_core") or {}).get("accounts") or []
    for index, account in enumerate(incoming_accounts):
        if not isinstance(account, dict):
            continue
        if not account.get("password") and index < len(old_accounts):
            account["password"] = old_accounts[index].get("password", "")

    # Admin credentials are managed through setup/login only.
    payload.pop("admin", None)

    merged = {**old, **payload}
    try:
        new_config = Config.model_validate(merged)
    except Exception as exc:  # noqa: BLE001
        raise _err(400, f"invalid config: {exc}") from exc

    new_config.admin = ctx.config.admin
    try:
        new_config.save(ctx.config_path)
    except OSError as exc:
        raise _err(500, f"failed to save config: {exc}") from exc

    ctx.config = new_config
    logger.info("config updated via admin API (restart required for account changes)")
    return JSONResponse(_masked_config(new_config))


async def logs(request: Request, ctx) -> JSONResponse:
    limit = int(request.query_params.get("limit", 50))
    return JSONResponse({"logs": ctx.stats.recent_logs(limit)})


async def runtime_logs(request: Request, ctx) -> JSONResponse:
    offset = int(request.query_params.get("offset", 0))
    limit = int(request.query_params.get("limit", 100))
    entries = ctx.runtime_logs[offset : offset + limit] if ctx.runtime_logs else []
    return JSONResponse(
        {"total": len(ctx.runtime_logs or []), "offset": offset, "limit": limit, "logs": entries}
    )


async def sessions(ctx) -> JSONResponse:
    # Upstream session listing is not implemented; the dashboard tolerates empty.
    return JSONResponse({"sessions": [], "next_updated_at": None})
