"""Message fingerprint chains for session reuse.

Reuse is only safe for plain conversational turns. Anything that changes how the
prompt is rendered (tools, tool results, inline files, remote images,
``response_format``) forces a fresh session with the full prompt - this mirrors
``request/delta.rs`` in the Rust project.
"""

from __future__ import annotations

import hashlib

from ..types import ChatCompletionsRequest
from .prompt import format_content


def _fingerprint(message) -> str:
    content = format_content(message.content)
    if message.tool_calls:
        content += "|" + "|".join(
            f"{call.function.name}:{call.function.arguments}" for call in message.tool_calls
        )
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{message.role}:{digest}"


def build(req: ChatCompletionsRequest) -> tuple[str, ...] | None:
    """Return the chain, or ``None`` when reuse must be disabled."""
    if req.tools or req.response_format is not None:
        return None

    chain: list[str] = []
    for message in req.messages:
        if message.role in ("tool", "function"):
            return None
        if isinstance(message.content, list):
            for part in message.content:
                if part.type == "file":
                    return None
                if part.type == "image_url" and part.image_url:
                    if part.image_url.url.startswith(("http://", "https://")):
                        return None
        chain.append(_fingerprint(message))

    if not chain:
        return None
    return tuple(chain)


def incremental_prompt(req: ChatCompletionsRequest) -> str | None:
    """The text to send when a cached session is reused: the last user message only."""
    for message in reversed(req.messages):
        if message.role == "user":
            return format_content(message.content)
    return None
