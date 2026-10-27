"""Upstream patch protocol (``p`` / ``o`` / ``v``).

Port of ``apply_patch`` in ``ds_core/src/chat/response.rs``.

The upstream stream is not "one event = one text delta"; it is a patch stream:

    data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", ...}]}
    data: {"p": "response/fragments/-1/content", "v": "next chunk"}
    data: {"p": "response/status", "v": "FINISHED"}

``p`` and ``o`` persist across frames, ``o`` defaults to ``SET``, and ``BATCH``
carries an array of sub-patches whose paths are prefixed with the parent path.
"""

from __future__ import annotations

from typing import Any

from .events import ContentDelta, StreamEvent, ThinkDelta

FRAG_THINK = "THINK"
FRAG_RESPONSE = "RESPONSE"

_STATUS_PATHS = frozenset({"response/status", "/response/status"})
_USAGE_PATHS = frozenset(
    {
        "response/accumulated_token_usage",
        "accumulated_token_usage",
        "/response/accumulated_token_usage",
        "/accumulated_token_usage",
    }
)
_LAST_CONTENT_PATHS = frozenset(
    {"response/fragments/-1/content", "/response/fragments/-1/content"}
)
_FRAGMENTS_PATHS = frozenset({"response/fragments", "/response/fragments"})


class PatchParser:
    """Stateful parser for one upstream stream."""

    __slots__ = (
        "current_path",
        "current_op",
        "status",
        "accumulated_token_usage",
        "fragments",
        "response_has_content",
    )

    def __init__(self) -> None:
        self.current_path: str | None = None
        self.current_op: str | None = None
        self.status: str | None = None
        self.accumulated_token_usage: int | None = None
        self.fragments: list[str] = []
        self.response_has_content = False

    def apply(self, payload: dict[str, Any]) -> list[StreamEvent]:
        if not isinstance(payload, dict):
            return []

        path_value = payload.get("p")
        if isinstance(path_value, str):
            self.current_path = path_value
        op_value = payload.get("o")
        if isinstance(op_value, str):
            self.current_op = op_value

        op = self.current_op or "SET"
        path = self.current_path or ""

        if "v" not in payload:
            return []
        value = payload["v"]

        # Initial snapshot: no path yet and the value carries a whole `response`.
        if self.current_path is None and isinstance(value, dict) and "response" in value:
            return self._apply_initial_snapshot(value["response"])

        if op == "BATCH" and isinstance(value, list):
            return self._apply_batch(path, value)

        return self._apply_path(path, op, value)

    # -- internals ---------------------------------------------------------

    def _apply_batch(self, parent_path: str, items: list[Any]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        sub_path = ""
        sub_op = "SET"

        for item in items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("p"), str):
                sub_path = item["p"]
            if isinstance(item.get("o"), str):
                sub_op = item["o"]
            if "v" not in item:
                continue
            value = item["v"]

            if not parent_path:
                full_path = sub_path
            elif not sub_path:
                full_path = parent_path
            else:
                full_path = f"{parent_path}/{sub_path}"

            if sub_op == "BATCH":
                events.extend(self._apply_batch(full_path, value if isinstance(value, list) else []))
            else:
                events.extend(self._apply_path(full_path, sub_op, value))
        return events

    def _apply_initial_snapshot(self, response: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if not isinstance(response, dict):
            return events

        status = response.get("status")
        if isinstance(status, str):
            self.status = status

        usage = response.get("accumulated_token_usage")
        if isinstance(usage, int):
            self.accumulated_token_usage = min(max(usage, 0), 0xFFFFFFFF)

        fragments = response.get("fragments")
        if isinstance(fragments, list):
            self.fragments.clear()
            for frag in fragments:
                if not isinstance(frag, dict):
                    continue
                frag_type = frag.get("type")
                if not isinstance(frag_type, str):
                    continue
                content = frag.get("content")
                content = content if isinstance(content, str) else ""
                events.extend(self._emit(frag_type, content))
                self.fragments.append(frag_type)
        return events

    def _apply_path(self, path: str, op: str, value: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if path in _STATUS_PATHS:
            if isinstance(value, str):
                self.status = value
            return events

        if path in _USAGE_PATHS:
            if isinstance(value, int):
                self.accumulated_token_usage = min(max(value, 0), 0xFFFFFFFF)
            return events

        if path in _LAST_CONTENT_PATHS:
            if isinstance(value, str) and self.fragments:
                events.extend(self._emit(self.fragments[-1], value))
            return events

        if path in _FRAGMENTS_PATHS and op == "APPEND" and isinstance(value, list):
            for frag in value:
                if not isinstance(frag, dict):
                    continue
                frag_type = frag.get("type")
                if not isinstance(frag_type, str):
                    continue
                content = frag.get("content")
                content = content if isinstance(content, str) else ""
                events.extend(self._emit(frag_type, content))
                self.fragments.append(frag_type)
            return events

        return events

    def _emit(self, frag_type: str, content: str) -> list[StreamEvent]:
        if not content:
            return []
        if frag_type == FRAG_THINK:
            return [ThinkDelta(content)]
        if frag_type == FRAG_RESPONSE:
            self.response_has_content = True
            return [ContentDelta(content)]
        return []
