"""Local ``stop`` enforcement.

The upstream has no ``stop`` parameter, so we truncate locally. A stop sequence can
straddle two chunks, so the detector always holds back ``max_len - 1`` characters.
"""

from __future__ import annotations


class StopDetector:
    __slots__ = ("stops", "_buf", "_hold", "hit")

    def __init__(self, stops: list[str]) -> None:
        self.stops = [s for s in stops if s]
        self._buf = ""
        self._hold = max((len(s) for s in self.stops), default=0)
        self.hit = False

    def feed(self, text: str) -> str:
        """Return the part of ``text`` that is safe to emit."""
        if not self.stops or self.hit or not text:
            return text if not self.hit else ""

        self._buf += text
        for stop in self.stops:
            index = self._buf.find(stop)
            if index >= 0:
                self.hit = True
                out = self._buf[:index]
                self._buf = ""
                return out

        if len(self._buf) > self._hold - 1:
            cut = len(self._buf) - (self._hold - 1)
            out = self._buf[:cut]
            self._buf = self._buf[cut:]
            return out
        return ""

    def flush(self) -> str:
        """Emit whatever is still held back (only when no stop was hit)."""
        if self.hit:
            self._buf = ""
            return ""
        out = self._buf
        self._buf = ""
        return out
