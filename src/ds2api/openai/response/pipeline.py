"""Streaming pipeline: upstream events -> OpenAI chunks -> SSE bytes.

Keepalive note: ``asyncio.wait_for(anext(gen), 1)`` cancels the inner generator on
timeout and kills the upstream socket (verified in Phase 0). The pattern below
holds one pending ``anext`` task and races it against a timer instead, which keeps
the generator alive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ...core.events import (
    ContentDelta,
    Done,
    Meta,
    StreamEvent,
    ThinkDelta,
)
from .converter import KEEPALIVE_ID, ChunkFactory, usage_dict
from .stop_detect import StopDetector
from .tool_parser import TagConfig
from .tool_stream import CallsPiece, ContentPiece, ToolStream

logger = logging.getLogger("ds2api.pipeline")

_KEEPALIVE = object()


async def _with_keepalive(
    events: AsyncIterator[StreamEvent], interval: float
) -> AsyncIterator[Any]:
    """Yield upstream events, emitting ``_KEEPALIVE`` when the source is idle."""
    iterator = events.__aiter__()
    task: asyncio.Future | None = None
    try:
        while True:
            if task is None:
                task = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({task}, timeout=interval)
            if task in done:
                try:
                    event = task.result()
                except StopAsyncIteration:
                    task = None
                    break
                finally:
                    if task is not None and task.done():
                        task = None
                yield event
            else:
                yield _KEEPALIVE
    finally:
        if task is not None and not task.done():
            task.cancel()


async def iter_chunks(
    events: AsyncIterator[StreamEvent],
    *,
    model: str,
    prompt_tokens: int,
    stop: list[str] | None = None,
    include_obfuscation: bool = True,
    cfg: TagConfig | None = None,
    keepalive_interval: float = 1.0,
) -> AsyncIterator[dict]:
    factory = ChunkFactory(model, include_obfuscation=include_obfuscation)
    tool_stream = ToolStream(cfg)
    detector = StopDetector(stop or [])
    finish_reason: str | None = None
    completion_tokens = 0
    stopped = False
    done_seen = False
    upstream_finish: str | None = None

    def emit_content(text: str) -> list[dict]:
        out: list[dict] = []
        safe = detector.feed(text)
        if safe:
            out.append(factory.chunk({"content": safe}))
        return out

    async for event in _with_keepalive(events, keepalive_interval):
        if event is _KEEPALIVE:
            yield factory.keepalive()
            continue

        if isinstance(event, Meta):
            # First frame carries the role and a prompt-only usage block.
            yield factory.chunk(
                {"role": "assistant", "content": ""},
                usage=usage_dict(prompt_tokens, 0),
            )
            continue

        if isinstance(event, ThinkDelta):
            if stopped:
                continue
            if event.content:
                yield factory.chunk({"reasoning_content": event.content})
            continue

        if isinstance(event, ContentDelta):
            if stopped:
                continue
            for piece in tool_stream.feed(event.content):
                if isinstance(piece, ContentPiece):
                    for chunk in emit_content(piece.text):
                        yield chunk
                    if detector.hit:
                        stopped = True
                elif isinstance(piece, CallsPiece):
                    finish_reason = "tool_calls"
                    yield factory.chunk(
                        {"tool_calls": [call.to_openai() for call in piece.calls]}
                    )
            continue

        if isinstance(event, Done):
            done_seen = True
            upstream_finish = event.finish_reason
            completion_tokens = event.accumulated_token_usage or 0
            break

    # Flush whatever is still buffered, whether the stream ended cleanly or not.
    for piece in tool_stream.finish():
        if isinstance(piece, ContentPiece):
            for chunk in emit_content(piece.text):
                yield chunk
        elif isinstance(piece, CallsPiece):
            finish_reason = "tool_calls"
            yield factory.chunk({"tool_calls": [call.to_openai() for call in piece.calls]})

    tail = detector.flush()
    if tail:
        yield factory.chunk({"content": tail})

    if finish_reason is None:
        if upstream_finish == "length":
            finish_reason = "length"
        elif done_seen:
            finish_reason = "stop"

    yield factory.chunk(
        {},
        finish_reason=finish_reason,
        usage=usage_dict(prompt_tokens, completion_tokens),
    )


def sse_bytes(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


async def stream_sse(
    events: AsyncIterator[StreamEvent],
    *,
    model: str,
    prompt_tokens: int,
    stop: list[str] | None = None,
    include_obfuscation: bool = True,
    cfg: TagConfig | None = None,
    keepalive_interval: float = 1.0,
) -> AsyncIterator[bytes]:
    """Serialise chunks and append exactly one ``[DONE]`` on clean completion."""
    async for chunk in iter_chunks(
        events,
        model=model,
        prompt_tokens=prompt_tokens,
        stop=stop,
        include_obfuscation=include_obfuscation,
        cfg=cfg,
        keepalive_interval=keepalive_interval,
    ):
        yield sse_bytes(chunk)
    yield b"data: [DONE]\n\n"


async def aggregate(
    events: AsyncIterator[StreamEvent],
    *,
    model: str,
    prompt_tokens: int,
    stop: list[str] | None = None,
    cfg: TagConfig | None = None,
) -> dict:
    """Collect the whole stream into a single ``chat.completion`` object."""
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    usage: dict | None = None
    created = None
    completion_id = None

    async for chunk in iter_chunks(
        events,
        model=model,
        prompt_tokens=prompt_tokens,
        stop=stop,
        include_obfuscation=False,
        cfg=cfg,
    ):
        if chunk.get("id") == KEEPALIVE_ID:
            continue
        if completion_id is None:
            completion_id = chunk.get("id")
            created = chunk.get("created")
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if delta.get("tool_calls"):
                tool_calls = delta["tool_calls"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content) if content else ("" if tool_calls else ""),
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": completion_id or "chatcmpl-0",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }
