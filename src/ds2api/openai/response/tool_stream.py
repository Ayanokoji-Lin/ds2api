"""Streaming tool-call detection.

The model emits tool calls as text, so the parser has to decide, while bytes are
still arriving, whether a chunk is prose or the start of a DSML block. Three
states:

    DETECTING   buffer, look for an opener, release everything except a safety tail
    COLLECTING  hold the whole block until it closes (or hits the size cap)
    DONE        a call was emitted; drop the rest of the text but keep the stream open

The safety tail must be at least as long as the longest tag we might match, in
bytes. ``TAG_WINDOW`` is 128 for DSML (the Rust project uses 71 for its shorter
legacy tags).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import dsml
from .tags import DSML_CLOSE_RE, MAX_XML_BUF_LEN, TAG_WINDOW, normalize_tags
from .tool_parser import TagConfig, ToolCallOut, detect_start, parse_tool_calls

logger = logging.getLogger("ds2api.tool_stream")

DETECTING = "detecting"
COLLECTING = "collecting"
DONE = "done"


@dataclass(slots=True)
class ContentPiece:
    text: str


@dataclass(slots=True)
class CallsPiece:
    calls: list[ToolCallOut]


Piece = ContentPiece | CallsPiece


def _is_complete(buffer: str, kind: str, cfg: TagConfig) -> bool:
    if kind == "dsml":
        for match in DSML_CLOSE_RE.finditer(buffer):
            if match.group(1) in dsml.DSML_CALLS_NAMES:
                return True
        return False
    norm = normalize_tags(buffer)
    return any(normalize_tags(end) in norm for end in cfg.ends)


class ToolStream:
    """Turn a stream of text deltas into content/calls pieces."""

    __slots__ = ("cfg", "_buffer", "_state", "_kind", "_emitted_calls")

    def __init__(self, cfg: TagConfig | None = None) -> None:
        self.cfg = cfg or TagConfig()
        self._buffer = ""
        self._state = DETECTING
        self._kind = ""
        self._emitted_calls = False

    @property
    def state(self) -> str:
        return self._state

    def feed(self, text: str) -> list[Piece]:
        if self._state == DONE or not text:
            return []
        self._buffer += text
        return self._advance()

    def finish(self) -> list[Piece]:
        """Flush at end-of-stream."""
        if self._state == DONE:
            self._buffer = ""
            return []
        if self._state == DETECTING:
            out = [ContentPiece(self._buffer)] if self._buffer else []
            self._buffer = ""
            return out
        # COLLECTING: the block never closed. Try one last parse anyway.
        pieces = self._try_parse(final=True)
        if pieces is None:
            out = [ContentPiece(self._buffer)] if self._buffer else []
            self._buffer = ""
            self._state = DONE
            return out
        return pieces

    # -- internals ---------------------------------------------------------

    def _advance(self) -> list[Piece]:
        if self._state == DETECTING:
            found = detect_start(self._buffer)
            if found is None:
                return self._release_safe_tail()
            offset, kind = found
            pieces: list[Piece] = []
            if offset > 0:
                pieces.append(ContentPiece(self._buffer[:offset]))
            self._buffer = self._buffer[offset:]
            self._kind = kind
            self._state = COLLECTING
            pieces.extend(self._collect())
            return pieces

        return self._collect()

    def _release_safe_tail(self) -> list[Piece]:
        if len(self._buffer) <= TAG_WINDOW:
            return []
        cut = len(self._buffer) - TAG_WINDOW
        out = self._buffer[:cut]
        self._buffer = self._buffer[cut:]
        return [ContentPiece(out)] if out else []

    def _collect(self) -> list[Piece]:
        if len(self._buffer) > MAX_XML_BUF_LEN:
            logger.warning("tool-call block exceeded %d bytes; falling back to text", MAX_XML_BUF_LEN)
            out = [ContentPiece(self._buffer)]
            self._buffer = ""
            self._state = DONE
            return out

        if not _is_complete(self._buffer, self._kind, self.cfg):
            return []

        pieces = self._try_parse(final=False)
        if pieces is None:
            return []
        return pieces

    def _try_parse(self, *, final: bool) -> list[Piece] | None:
        buffer = self._buffer
        result = parse_tool_calls(buffer, self.cfg)
        if result is None:
            if final:
                return None
            # Not parseable yet (e.g. the JSON array is still incomplete).
            return None
        calls, remaining = result
        self._buffer = ""
        self._state = DONE
        self._emitted_calls = True
        pieces: list[Piece] = []
        if remaining:
            pieces.append(ContentPiece(remaining))
        pieces.append(CallsPiece(calls))
        return pieces
