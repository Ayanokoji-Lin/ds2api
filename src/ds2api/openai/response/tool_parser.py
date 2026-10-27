"""Tool-call parsing: DSML first, then legacy tags, then bare JSON, then XML.

The dispatcher order matters. DSML is the primary dialect for DeepSeek V4+, but
older models still emit the ``<|tool▁calls▁begin|>`` block, and some outputs are
just a bare JSON array. All four paths converge on the same ``ToolCallOut`` shape.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ...core.errors import BadRequest
from . import dsml
from .tags import (
    MAX_XML_BUF_LEN,
    TOOL_CALL_END,
    TOOL_CALL_START,
    is_inside_code_fence,
    normalize_tags,
)

logger = logging.getLogger("ds2api.tool_parser")


@dataclass(slots=True)
class ToolCallOut:
    id: str
    name: str
    arguments: str
    index: int
    namespace: str | None = None

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
            "index": self.index,
        }


@dataclass(slots=True)
class TagConfig:
    starts: list[str] = field(default_factory=lambda: [TOOL_CALL_START])
    ends: list[str] = field(default_factory=lambda: [TOOL_CALL_END])


_counter = {"n": 0}


def next_call_id() -> str:
    _counter["n"] += 1
    return f"call_{_counter['n']:016x}"


def reset_call_ids() -> None:
    _counter["n"] = 0


# -- JSON repair -----------------------------------------------------------

_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')


def repair_json(text: str) -> str:
    """Two fixes, in this order: invalid backslashes, then bare keys."""
    text = _BAD_ESCAPE.sub(r"\\\\", text)
    return _BARE_KEY.sub(r'\1"\2":', text)


def _loads(raw: str) -> object | None:
    for candidate in (raw, repair_json(raw)):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


# -- converters ------------------------------------------------------------


def _from_parsed(parsed: list[dsml.ParsedCall]) -> list[ToolCallOut]:
    return [
        ToolCallOut(
            id=next_call_id(),
            name=call.name,
            arguments=call.arguments_json(),
            index=index,
            namespace=call.namespace,
        )
        for index, call in enumerate(parsed)
    ]


def _from_json_value(value: object) -> list[ToolCallOut] | None:
    """Convert a parsed JSON array/object of ``{name, arguments}`` items."""
    items: list[object]
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = [value]
    else:
        return None

    calls: list[ToolCallOut] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or (item.get("function") or {}).get("name")
        if not isinstance(name, str) or not name:
            return None
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            parsed_args = _loads(args)
            args = parsed_args if parsed_args is not None else {}
        if not isinstance(args, (dict, list)):
            args = {}
        calls.append(
            ToolCallOut(
                id=next_call_id(),
                name=name,
                arguments=json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                index=len(calls),
            )
        )
    return calls or None


_XML_INVOKE = re.compile(r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</invoke>", re.S)
_XML_PARAM = re.compile(
    r"<parameter\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</parameter>", re.S
)


def _parse_xml_invoke(text: str) -> list[ToolCallOut] | None:
    calls: list[ToolCallOut] = []
    for match in _XML_INVOKE.finditer(text):
        name = match.group(1)
        params: dict[str, object] = {}
        for pname, pvalue in _XML_PARAM.findall(match.group(2)):
            parsed = _loads(pvalue.strip())
            params[pname] = parsed if parsed is not None else pvalue
        calls.append(
            ToolCallOut(
                id=next_call_id(),
                name=name,
                arguments=json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                index=len(calls),
            )
        )
    return calls or None


# -- legacy tag block ------------------------------------------------------


def _find_legacy_block(text: str, cfg: TagConfig) -> tuple[int, int, int] | None:
    """Return ``(start, inner_start, inner_end)`` for the first legacy block."""
    norm = normalize_tags(text)
    best: tuple[int, int] | None = None
    for start_tag in cfg.starts:
        needle = normalize_tags(start_tag)
        pos = norm.find(needle)
        if pos < 0:
            continue
        if best is None or pos < best[0]:
            best = (pos, pos + len(needle))
    if best is None:
        return None

    start, inner_start = best
    inner_end = len(text)

    # Closing candidates: configured ends, plus a second opening tag.
    for end_tag in cfg.ends:
        needle = normalize_tags(end_tag)
        pos = norm.find(needle, inner_start)
        if pos >= 0 and pos < inner_end:
            inner_end = pos
    for start_tag in cfg.starts:
        needle = normalize_tags(start_tag)
        pos = norm.find(needle, inner_start)
        if pos >= 0 and pos < inner_end:
            inner_end = pos

    return start, inner_start, inner_end


def _parse_legacy(text: str, cfg: TagConfig) -> tuple[list[ToolCallOut], str] | None:
    located = _find_legacy_block(text, cfg)
    if located is None:
        return None
    start, inner_start, inner_end = located
    # A tag inside a markdown code fence is illustrative, not a real call.
    if is_inside_code_fence(text, start):
        return None
    inner = text[inner_start:inner_end]

    # Prefer a JSON array; fall back to a single object, then to XML.
    calls: list[ToolCallOut] | None = None
    bracket = inner.find("[")
    if bracket >= 0:
        closing = inner.rfind("]")
        if closing > bracket:
            value = _loads(inner[bracket : closing + 1])
            if value is not None:
                calls = _from_json_value(value)
    if calls is None:
        brace = inner.find("{")
        if brace >= 0:
            closing = inner.rfind("}")
            if closing > brace:
                value = _loads(inner[brace : closing + 1])
                if value is not None:
                    calls = _from_json_value(value)
    if calls is None:
        calls = _parse_xml_invoke(inner)
    if calls is None:
        return None

    # Remove everything from the opening tag through the closing tag.
    tail_start = inner_end
    for end_tag in cfg.ends:
        needle = normalize_tags(end_tag)
        pos = normalize_tags(text).find(needle, inner_start)
        if pos >= 0:
            tail_start = max(tail_start, pos + len(needle))
            break
    remaining = text[:start] + text[tail_start:]
    return calls, remaining


# -- bare JSON -------------------------------------------------------------


def _parse_bare_json(text: str) -> tuple[list[ToolCallOut], str] | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    value = _loads(stripped)
    if value is None:
        return None
    calls = _from_json_value(value)
    if calls is None:
        return None
    return calls, ""


# -- public entry ----------------------------------------------------------


def parse_tool_calls(
    text: str, cfg: TagConfig | None = None
) -> tuple[list[ToolCallOut], str] | None:
    """Parse tool calls out of a complete text block.

    Returns ``(calls, remaining_text)`` or ``None`` when there is nothing to parse.
    """
    cfg = cfg or TagConfig()
    if not text:
        return None

    start = dsml.find_calls_start(text)
    if start is not None and not is_inside_code_fence(text, start):
        parsed = dsml.parse_dsml(text)
        if parsed is not None:
            calls, remaining = parsed
            return _from_parsed(calls), remaining

    if not is_inside_code_fence(text, 0):
        legacy = _parse_legacy(text, cfg)
        if legacy is not None:
            return legacy

    xml = _parse_xml_invoke(text)
    if xml is not None:
        return xml, ""

    return _parse_bare_json(text)


def detect_start(text: str, cfg: TagConfig | None = None) -> tuple[int, str] | None:
    """Locate the earliest tool-call opener.

    Returns ``(offset, kind)`` where kind is ``"dsml"`` or ``"legacy"``.
    """
    cfg = cfg or TagConfig()
    norm = normalize_tags(text)

    candidates: list[tuple[int, str]] = []
    match = dsml.DSML_OPEN_RE.search(norm)
    while match is not None:
        if match.group(1) in dsml.DSML_CALLS_NAMES:
            candidates.append((match.start(), "dsml"))
            break
        match = dsml.DSML_OPEN_RE.search(norm, match.end())

    for start_tag in cfg.starts:
        pos = norm.find(normalize_tags(start_tag))
        if pos >= 0:
            candidates.append((pos, "legacy"))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


__all__ = [
    "MAX_XML_BUF_LEN",
    "TagConfig",
    "ToolCallOut",
    "detect_start",
    "next_call_id",
    "parse_tool_calls",
    "repair_json",
    "reset_call_ids",
    "BadRequest",
]
