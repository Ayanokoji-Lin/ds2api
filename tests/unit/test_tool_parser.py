"""Tool-call parser tests across all four dialects."""

from __future__ import annotations

import json

import pytest

from ds2api.openai.response.tool_parser import (
    TagConfig,
    detect_start,
    parse_tool_calls,
    repair_json,
    reset_call_ids,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_call_ids()


def test_dsml_is_parsed() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="f">'
        '<｜DSML｜ parameter name="a" string="false">1</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    result = parse_tool_calls(text)
    assert result is not None
    calls, remaining = result
    assert len(calls) == 1
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"a": 1}
    assert remaining == ""


def test_legacy_tag_block() -> None:
    text = (
        "sure\n<|tool▁calls▁begin|>\n"
        '[{"name": "get_weather", "arguments": {"city": "SH"}}]\n'
        "<|tool▁calls▁end|>"
    )
    result = parse_tool_calls(text)
    assert result is not None
    calls, remaining = result
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "SH"}
    assert "sure" in remaining
    assert "tool" not in remaining


def test_legacy_fuzzy_tag_with_ascii_underscores() -> None:
    # ``▁`` (U+2581) and ``_`` are interchangeable in the legacy protocol.
    text = (
        "<|tool_calls_begin|>"
        '[{"name": "f", "arguments": {}}]'
        "<|tool_calls_end|>"
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert result[0][0].name == "f"


def test_bare_json_array() -> None:
    text = '[{"name": "f", "arguments": {"x": 2}}]'
    result = parse_tool_calls(text)
    assert result is not None
    calls, remaining = result
    assert calls[0].name == "f"
    assert json.loads(calls[0].arguments) == {"x": 2}
    assert remaining == ""


def test_xml_invoke_fallback() -> None:
    text = '<invoke name="search"><parameter name="q">hello</parameter></invoke>'
    result = parse_tool_calls(text)
    assert result is not None
    calls, _ = result
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"q": "hello"}


def test_multiple_calls_get_distinct_ids() -> None:
    text = (
        '<｜DSML｜ calls>'
        '<｜DSML｜ invoke name="a"><｜DSML｜ parameter name="x" string="false">1</｜DSML｜ parameter></｜DSML｜ invoke>'
        '<｜DSML｜ invoke name="b"><｜DSML｜ parameter name="y" string="false">2</｜DSML｜ parameter></｜DSML｜ invoke>'
        "</｜DSML｜ calls>"
    )
    result = parse_tool_calls(text)
    assert result is not None
    calls, _ = result
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].id != calls[1].id
    assert [c.index for c in calls] == [0, 1]


def test_code_fence_is_ignored() -> None:
    text = "```\n<|tool▁calls▁begin|>[{\"name\":\"f\"}]<|tool▁calls▁end|>\n```"
    assert parse_tool_calls(text) is None


def test_plain_text_returns_none() -> None:
    assert parse_tool_calls("nothing to see here") is None
    assert parse_tool_calls("") is None


def test_repair_bare_keys_and_bad_backslashes() -> None:
    broken = r'{name: "f", path: "C:\Users\x"}'
    repaired = repair_json(broken)
    value = json.loads(repaired)
    assert value["name"] == "f"


def test_arguments_are_compacted_and_sorted() -> None:
    text = '[{"name": "f", "arguments": {"b": 2, "a": 1}}]'
    result = parse_tool_calls(text)
    assert result is not None
    assert result[0][0].arguments == '{"a":1,"b":2}'


def test_detect_start_prefers_earliest() -> None:
    text = "hi <|tool▁calls▁begin|>[]"
    found = detect_start(text)
    assert found is not None
    assert found[1] == "legacy"

    dsml_text = "hi <｜DSML｜ calls></｜DSML｜ calls>"
    found = detect_start(dsml_text)
    assert found is not None
    assert found[1] == "dsml"
    assert found[0] == 3


def test_custom_tag_config() -> None:
    cfg = TagConfig(starts=["<tool>"], ends=["</tool>"])
    text = '<tool>[{"name": "f", "arguments": {}}]</tool>'
    result = parse_tool_calls(text, cfg)
    assert result is not None
    assert result[0][0].name == "f"
