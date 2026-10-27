"""Patch-protocol parser tests, driven by frames captured from the real upstream."""

from __future__ import annotations

import json

from ds2api.core.events import ContentDelta, ThinkDelta
from ds2api.core.patch import PatchParser


def feed(parser: PatchParser, *payloads: dict) -> list:
    out = []
    for payload in payloads:
        out.extend(parser.apply(payload))
    return out


def test_append_fragment_then_content_delta() -> None:
    parser = PatchParser()
    events = feed(
        parser,
        {"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": "想"}]},
        {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "答"}]},
        {"p": "response/fragments/-1/content", "v": "案"},
    )
    assert events == [ThinkDelta("想"), ContentDelta("答"), ContentDelta("案")]
    assert parser.response_has_content is True


def test_p_and_o_persist_across_frames() -> None:
    parser = PatchParser()
    events = feed(
        parser,
        {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "a"}]},
        # no p / o: both persist -> same APPEND on the same path
        {"v": [{"type": "RESPONSE", "content": "b"}]},
    )
    assert events == [ContentDelta("a"), ContentDelta("b")]


def test_default_op_is_set() -> None:
    parser = PatchParser()
    events = feed(parser, {"p": "response/status", "v": "FINISHED"})
    assert events == []
    assert parser.status == "FINISHED"


def test_batch_with_nested_paths() -> None:
    parser = PatchParser()
    events = feed(
        parser,
        {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "x"}]},
        {
            "p": "response",
            "o": "BATCH",
            "v": [
                {"p": "accumulated_token_usage", "v": 139},
                {"p": "quasi_status", "v": "FINISHED"},
            ],
        },
        {"p": "response/status", "o": "SET", "v": "FINISHED"},
    )
    assert events == [ContentDelta("x")]
    assert parser.accumulated_token_usage == 139
    assert parser.status == "FINISHED"


def test_nested_batch_recursion() -> None:
    parser = PatchParser()
    feed(
        parser,
        {
            "p": "response",
            "o": "BATCH",
            "v": [
                {
                    "o": "BATCH",
                    "v": [{"p": "accumulated_token_usage", "v": 7}],
                }
            ],
        },
    )
    assert parser.accumulated_token_usage == 7


def test_initial_snapshot() -> None:
    parser = PatchParser()
    events = feed(
        parser,
        {
            "v": {
                "response": {
                    "status": "WIP",
                    "accumulated_token_usage": 12,
                    "fragments": [
                        {"type": "THINK", "content": "think"},
                        {"type": "RESPONSE", "content": "resp"},
                        {"type": "TOOL_SEARCH", "content": "ignored"},
                    ],
                }
            }
        },
    )
    assert events == [ThinkDelta("think"), ContentDelta("resp")]
    assert parser.status == "WIP"
    assert parser.accumulated_token_usage == 12
    assert parser.fragments == ["THINK", "RESPONSE", "TOOL_SEARCH"]


def test_content_delta_without_fragment_is_ignored() -> None:
    parser = PatchParser()
    assert feed(parser, {"p": "response/fragments/-1/content", "v": "orphan"}) == []


def test_usage_is_clamped() -> None:
    parser = PatchParser()
    feed(parser, {"p": "response/accumulated_token_usage", "v": 2**40})
    assert parser.accumulated_token_usage == 0xFFFFFFFF


def test_non_dict_payload_ignored() -> None:
    parser = PatchParser()
    assert parser.apply([1, 2, 3]) == []  # type: ignore[arg-type]
