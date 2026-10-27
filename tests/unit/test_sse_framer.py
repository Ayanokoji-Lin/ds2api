"""SSE framing tests, including the truncated-tail case from issue #93."""

from __future__ import annotations

from ds2api.core.sse_framer import SseFramer, parse_frame


def test_splits_on_blank_line() -> None:
    framer = SseFramer()
    frames = framer.feed(b'event: ready\ndata: {"a":1}\n\ndata: {"b":2}\n\n')
    assert frames == [b'event: ready\ndata: {"a":1}', b'data: {"b":2}']
    assert framer.pending == 0


def test_partial_frame_is_buffered_until_complete() -> None:
    framer = SseFramer()
    assert framer.feed(b'data: {"a"') == []
    assert framer.feed(b':1}\n') == []
    assert framer.feed(b"\n") == [b'data: {"a":1}']


def test_flush_returns_unterminated_tail() -> None:
    framer = SseFramer()
    assert framer.feed(b'data: {"status":"FINISHED"}') == []
    assert framer.flush() == [b'data: {"status":"FINISHED"}']
    assert framer.flush() == []


def test_flush_ignores_whitespace_only_tail() -> None:
    framer = SseFramer()
    framer.feed(b"\n  ")
    assert framer.flush() == []


def test_parse_frame_event_and_data() -> None:
    assert parse_frame(b'event: ready\ndata: {"x":1}') == ("ready", '{"x":1}')


def test_parse_frame_multiline_data_joined() -> None:
    event, data = parse_frame(b"data: a\ndata: b")
    assert event is None
    assert data == "a\nb"


def test_parse_frame_ignores_comments() -> None:
    assert parse_frame(b": keep-alive") == (None, None)
