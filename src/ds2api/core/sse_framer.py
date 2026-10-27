"""SSE framing: split the upstream byte stream on ``\\n\\n``.

The upstream stream has no ``[DONE]`` marker; the last frame often arrives without
a trailing blank line, so ``flush()`` must be called at EOF or the final
``accumulated_token_usage`` / ``response/status`` frame is lost (issue #93 in the
Rust project).
"""

from __future__ import annotations

SEP = b"\n\n"


class SseFramer:
    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Append ``data`` and return every complete frame."""
        self._buf.extend(data)
        frames: list[bytes] = []
        while True:
            pos = self._buf.find(SEP)
            if pos < 0:
                break
            frames.append(bytes(self._buf[:pos]))
            del self._buf[: pos + len(SEP)]
        return frames

    def flush(self) -> list[bytes]:
        """Return any trailing frame that never got its blank-line terminator."""
        if not self._buf:
            return []
        frame = bytes(self._buf)
        self._buf.clear()
        return [frame] if frame.strip() else []

    @property
    def pending(self) -> int:
        return len(self._buf)


def parse_frame(frame: bytes) -> tuple[str | None, str | None]:
    """Split one SSE frame into ``(event, data)``.

    Multiple ``data:`` lines are joined with ``\\n`` per the SSE spec. Comment
    lines (``: keep-alive``) are ignored.
    """
    event: str | None = None
    data_lines: list[str] = []
    for raw_line in frame.split(b"\n"):
        line = raw_line.rstrip(b"\r")
        if not line or line.startswith(b":"):
            continue
        if line.startswith(b"event:"):
            event = line[6:].strip().decode("utf-8", "replace")
        elif line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value.decode("utf-8", "replace"))
    data = "\n".join(data_lines) if data_lines else None
    return event, data
