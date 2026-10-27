"""OpenAI chunk construction.

Two behaviours differ from the official OpenAI service and are reproduced on
purpose because the Rust project does the same:

* every chunk carries a fresh ``id`` (``chatcmpl-{n:016x}``), not one id per stream
* the first chunk already carries ``usage`` (prompt tokens only)
"""

from __future__ import annotations

import base64
import os
import time

KEEPALIVE_ID = "chatcmpl-keepalive"
OBFUSCATION_TARGET = 512
OBFUSCATION_MIN = 16


class ChunkFactory:
    __slots__ = ("model", "created", "_n", "include_obfuscation")

    def __init__(self, model: str, *, created: int | None = None, include_obfuscation: bool = True) -> None:
        self.model = model
        self.created = created if created is not None else int(time.time())
        self._n = 0
        self.include_obfuscation = include_obfuscation

    def next_id(self) -> str:
        self._n += 1
        return f"chatcmpl-{self._n:016x}"

    def chunk(
        self,
        delta: dict,
        *,
        finish_reason: str | None = None,
        usage: dict | None = None,
        chunk_id: str | None = None,
    ) -> dict:
        payload: dict = {
            "id": chunk_id or self.next_id(),
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        if self.include_obfuscation:
            payload["obfuscation"] = _padding()
        return payload

    def keepalive(self) -> dict:
        return self.chunk({}, chunk_id=KEEPALIVE_ID)


def _padding() -> str:
    raw = os.urandom(OBFUSCATION_TARGET)
    encoded = base64.b64encode(raw).decode("ascii")
    return encoded[: max(OBFUSCATION_MIN, OBFUSCATION_TARGET // 2)]


def usage_dict(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
