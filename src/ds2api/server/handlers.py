"""Request handlers."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.chat import ChatRequest, FileUpload
from ..core.errors import CoreError
from ..core.events import Meta, StreamEvent
from ..openai import models as models_api
from ..openai.request import delta as delta_mod
from ..openai.request import files as files_mod
from ..openai.request import normalize, prompt, tools
from ..openai.request.resolver import resolve
from ..openai.response import pipeline
from ..openai.response.tool_parser import TagConfig
from ..openai.types import ChatCompletionsRequest
from .errors import ApiError, from_core, not_found

logger = logging.getLogger("ds2api.handlers")


def _tag_config(config) -> TagConfig:
    core = config.ds_core
    starts = ["<|tool▁calls▁begin|>", *core.tool_call.extra_starts]
    ends = ["<|tool▁calls▁end|>", *core.tool_call.extra_ends]
    return TagConfig(starts=starts, ends=ends)


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 - matches Rust's `.ok()`
        logger.warning("tiktoken unavailable; prompt_tokens reported as 0")
        return 0


async def _prepend(first: StreamEvent, rest):
    yield first
    async for item in rest:
        yield item


async def chat_completions(request: Request, state) -> StreamingResponse | JSONResponse:
    config = state.config
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, f"invalid JSON body: {exc}", "invalid_request_error", "bad_request") from exc

    try:
        req = ChatCompletionsRequest.model_validate(payload)
        req = normalize.apply(config, req)

        protocol = config.ds_core.tool_protocol
        tool_ctx = tools.extract(req, protocol=protocol)
        prompt_text = prompt.build(req, tool_ctx)
        uploads, has_http_urls = files_mod.extract(req)
        resolved = resolve(config, req, has_http_urls=has_http_urls)

        chain = delta_mod.build(req)
        incremental = delta_mod.incremental_prompt(req) if chain else None

        chat_req = ChatRequest(
            prompt=prompt_text,
            model_type=resolved.model_type,
            thinking_enabled=resolved.thinking_enabled,
            search_enabled=resolved.search_enabled,
            files=[FileUpload(f.filename, f.content_type, f.data) for f in uploads],
            delta=chain,
            incremental_prompt=incremental,
        )

        prompt_tokens = _estimate_tokens(prompt_text)
        stops = normalize.stop_sequences(req)
        tag_cfg = _tag_config(config)
        events = state.chat.complete(chat_req)

        # Peek the first event so the account header can be set before streaming.
        first = await anext(events)
        account_id = first.account_id if isinstance(first, Meta) else ""
        stream = _prepend(first, events)

        if req.stream:
            body = pipeline.stream_sse(
                stream,
                model=req.model,
                prompt_tokens=prompt_tokens,
                stop=stops,
                cfg=tag_cfg,
            )
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-ds-account": account_id,
            }
            return StreamingResponse(body, media_type="text/event-stream", headers=headers)

        result = await pipeline.aggregate(
            stream,
            model=req.model,
            prompt_tokens=prompt_tokens,
            stop=stops,
            cfg=tag_cfg,
        )
        return JSONResponse(result, headers={"x-ds-account": account_id})

    except ApiError:
        raise
    except CoreError as exc:
        raise from_core(exc) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_completions failed")
        raise ApiError(500, str(exc), "server_error", "internal_error") from exc


async def list_models(request: Request, state) -> JSONResponse:
    return JSONResponse(models_api.list_models(state.config))


async def get_model(model_id: str, state) -> JSONResponse:
    model = models_api.get_model(state.config, model_id)
    if model is None:
        raise not_found(f"model {model_id!r} not found")
    return JSONResponse(model)


async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
