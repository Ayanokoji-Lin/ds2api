"""DSML (DeepSeek Markup Language) tool-call parser.

DSML is DeepSeek V4's native tool-call dialect. Shape::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="search">
    <｜DSML｜ parameter name="queries" string="true">小柴胡 布洛芬</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

Rules that matter:

* V4.1 requires a space after the ``<｜DSML｜>`` token; V4 omitted it. Both are
  accepted, as are the ASCII (``<||DSML||``) and doubled-bar variants.
* ``string="true"`` means the value is the *raw* text between the tags - no
  unescaping, no JSON parsing.
* ``string="false"`` means the value is JSON (number / bool / null / array /
  object).
* Tool names may be namespace-qualified (``search::lookup``); callers get the
  bare name back.

Deliberately not regex-only: a parameter value may contain a bare ``<``, so the
value is delimited by locating the *closing tag sequence*, not the next ``<``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ...core.errors import BadRequest
from .tags import (
    DSML_ATTR_RE,
    DSML_CALLS_NAMES,
    DSML_CLOSE_RE,
    DSML_OPEN_RE,
    DSML_TAG_INVOKE,
    DSML_TAG_PARAMETER,
    split_qualified_name,
)

logger = logging.getLogger("ds2api.dsml")


@dataclass(slots=True)
class ParsedCall:
    """One ``invoke`` block, already normalised for the OpenAI shape."""

    name: str
    namespace: str | None
    arguments: dict

    def arguments_json(self) -> str:
        # sort_keys keeps golden tests stable; separators match OpenAI's compact form.
        return json.dumps(
            self.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def _parse_attrs(blob: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in DSML_ATTR_RE.finditer(blob):
        key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(4) or ""
        attrs[key] = value
    return attrs


def find_calls_start(text: str) -> int | None:
    """Return the offset of the first ``calls`` / ``tool_calls`` opening tag."""
    for match in DSML_OPEN_RE.finditer(text):
        if match.group(1) in DSML_CALLS_NAMES:
            return match.start()
    return None


def locate_calls_block(text: str) -> tuple[int, int] | None:
    """Locate the outermost ``calls`` block.

    Returns ``(start, end)`` where ``end`` is just past the closing tag. When the
    closing tag is missing (truncated stream) ``end`` is ``len(text)``.
    """
    start = find_calls_start(text)
    if start is None:
        return None
    close = None
    for match in DSML_CLOSE_RE.finditer(text, start):
        if match.group(1) in DSML_CALLS_NAMES:
            close = match
            break
    end = close.end() if close else len(text)
    return start, end


def _find_close(text: str, tag: str, from_pos: int) -> tuple[int, int] | None:
    for match in DSML_CLOSE_RE.finditer(text, from_pos):
        if match.group(1) == tag:
            return match.start(), match.end()
    return None


def _parse_parameters(block: str) -> dict:
    """Parse every ``parameter`` inside one invoke body."""
    params: dict = {}
    pos = 0
    while True:
        match = DSML_OPEN_RE.search(block, pos)
        if not match:
            break
        tag = match.group(1)
        if tag != DSML_TAG_PARAMETER:
            pos = match.end()
            continue
        attrs = _parse_attrs(match.group(2))
        name = attrs.get("name")
        value_start = match.end()
        close = _find_close(block, DSML_TAG_PARAMETER, value_start)
        if close is None:
            raw = block[value_start:]
            pos = len(block)
        else:
            raw = block[value_start : close[0]]
            pos = close[1]
        if not name:
            continue
        params[name] = _coerce(raw, attrs.get("string", "true"))
    return params


def _coerce(raw: str, string_flag: str) -> object:
    if string_flag.lower() == "false":
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("DSML parameter marked string=false but is not valid JSON")
            return raw
    return raw


def parse_calls_block(block: str) -> list[ParsedCall] | None:
    """Parse a block that starts with the ``calls`` tag.

    Returns ``None`` when the block contains no ``invoke`` at all, so callers can
    fall back to other formats instead of swallowing the text.
    """
    calls: list[ParsedCall] = []
    pos = 0
    while True:
        match = DSML_OPEN_RE.search(block, pos)
        if not match:
            break
        if match.group(1) != DSML_TAG_INVOKE:
            pos = match.end()
            continue
        attrs = _parse_attrs(match.group(2))
        qualified = attrs.get("name", "")
        body_start = match.end()
        close = _find_close(block, DSML_TAG_INVOKE, body_start)
        if close is None:
            body = block[body_start:]
            pos = len(block)
        else:
            body = block[body_start : close[0]]
            pos = close[1]

        if not qualified:
            # An invoke without a name is unusable; let the repair path handle it.
            return None
        try:
            namespace, bare = split_qualified_name(qualified)
        except BadRequest:
            raise
        calls.append(
            ParsedCall(name=bare, namespace=namespace, arguments=_parse_parameters(body))
        )

    return calls or None


def parse_dsml(text: str) -> tuple[list[ParsedCall], str] | None:
    """Parse a complete DSML block out of ``text``.

    Returns ``(calls, remaining_text)`` or ``None`` when no DSML block is present.
    ``remaining_text`` is the text with the whole block removed, so callers can
    still surface any surrounding prose to the user.
    """
    located = locate_calls_block(text)
    if located is None:
        return None
    start, end = located
    calls = parse_calls_block(text[start:end])
    if calls is None:
        return None
    remaining = text[:start] + text[end:]
    return calls, remaining


def looks_like_dsml(text: str) -> bool:
    """Cheap check used by the unified dispatcher."""
    return find_calls_start(text) is not None
