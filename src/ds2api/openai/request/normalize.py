"""Request validation and defaulting."""

from __future__ import annotations

from ...core.errors import BadRequest
from ...settings import Config
from ..types import ChatCompletionsRequest


def apply(config: Config, req: ChatCompletionsRequest) -> ChatCompletionsRequest:
    if not req.messages:
        raise BadRequest("messages must not be empty")

    if req.n is not None and req.n > 1:
        raise BadRequest("n > 1 is not supported by this gateway")

    for index, message in enumerate(req.messages):
        if not message.role:
            raise BadRequest(f"messages[{index}].role is required")
        if message.role == "tool" and not message.tool_call_id:
            raise BadRequest(f"messages[{index}].tool_call_id is required for tool messages")

    if not req.model:
        req.model = config.model_ids[0]

    return req


def stop_sequences(req: ChatCompletionsRequest) -> list[str]:
    """Normalise ``stop`` into a list of non-empty strings."""
    raw = req.stop
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    return [item for item in raw if item]
