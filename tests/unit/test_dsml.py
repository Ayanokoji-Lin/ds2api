"""DSML parser tests.

Covers the four known tag spellings, parameter typing, namespaces, multiple
invokes, and the tricky cases (bare ``<`` inside a value, missing close tag).
"""

from __future__ import annotations

import json

import pytest

from ds2api.core.errors import BadRequest
from ds2api.openai.response.dsml import parse_dsml, looks_like_dsml


def one_call(text: str):
    result = parse_dsml(text)
    assert result is not None, f"no DSML block parsed from {text!r}"
    calls, remaining = result
    assert len(calls) == 1
    return calls[0], remaining


# -- the four spellings ----------------------------------------------------


V41 = """<｜DSML｜ calls>
<｜DSML｜ invoke name="search">
<｜DSML｜ parameter name="queries" string="true">小柴胡 布洛芬</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""

V4 = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="queries" string="true">小柴胡 布洛芬</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

ASCII = """<||DSML||tool_calls>
<||DSML||invoke name="search">
<||DSML||parameter name="queries" string="true">小柴胡 布洛芬</||DSML||parameter>
</||DSML||invoke>
</||DSML||tool_calls>"""

DOUBLE_BAR = """<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="search">
<｜｜DSML｜｜parameter name="queries" string="true">小柴胡 布洛芬</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>"""


@pytest.mark.parametrize("text", [V41, V4, ASCII, DOUBLE_BAR])
def test_all_spellings_parse_identically(text: str) -> None:
    call, remaining = one_call(text)
    assert call.name == "search"
    assert call.namespace is None
    assert call.arguments == {"queries": "小柴胡 布洛芬"}
    assert remaining.strip() == ""


# -- parameter typing ------------------------------------------------------


def test_string_true_keeps_raw_text() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="echo">'
        '<｜DSML｜ parameter name="v" string="true">line1\nline2 "quoted" \\n</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    call, _ = one_call(text)
    assert call.arguments["v"] == 'line1\nline2 "quoted" \\n'


def test_string_false_parses_json_scalars() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="f">'
        '<｜DSML｜ parameter name="i" string="false">42</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="b" string="false">true</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="n" string="false">null</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="arr" string="false">[1,2,3]</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="obj" string="false">{"a":{"b":1}}</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    call, _ = one_call(text)
    assert call.arguments == {"i": 42, "b": True, "n": None, "arr": [1, 2, 3], "obj": {"a": {"b": 1}}}
    assert json.loads(call.arguments_json()) == call.arguments


def test_missing_string_attribute_defaults_to_true() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="f">'
        '<｜DSML｜ parameter name="n">123</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    call, _ = one_call(text)
    assert call.arguments["n"] == "123"


def test_bare_angle_bracket_inside_value() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="f">'
        '<｜DSML｜ parameter name="v" string="true">a < b and <tag> here</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    call, _ = one_call(text)
    assert call.arguments["v"] == "a < b and <tag> here"


# -- namespaces ------------------------------------------------------------


def test_namespace_is_split_off() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="search::lookup">'
        '<｜DSML｜ parameter name="q" string="true">x</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    call, _ = one_call(text)
    assert call.name == "lookup"
    assert call.namespace == "search"


def test_double_separator_rejected() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="a::b::c">'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    with pytest.raises(BadRequest):
        parse_dsml(text)


# -- multiple invokes ------------------------------------------------------


def test_multiple_invokes_preserve_order() -> None:
    text = """<｜DSML｜ calls>
<｜DSML｜ invoke name="a"><｜DSML｜ parameter name="x" string="false">1</｜DSML｜ parameter></｜DSML｜ invoke>
<｜DSML｜ invoke name="b"><｜DSML｜ parameter name="y" string="true">two</｜DSML｜ parameter></｜DSML｜ invoke>
<｜DSML｜ invoke name="c"><｜DSML｜ parameter name="z" string="false">[3]</｜DSML｜ parameter></｜DSML｜ invoke>
</｜DSML｜ calls>"""
    result = parse_dsml(text)
    assert result is not None
    calls, _ = result
    assert [c.name for c in calls] == ["a", "b", "c"]
    assert [c.arguments for c in calls] == [{"x": 1}, {"y": "two"}, {"z": [3]}]


# -- surrounding prose -----------------------------------------------------


def test_surrounding_text_is_preserved() -> None:
    text = "好的，我来查一下。\n" + V41 + "\n稍等。"
    call, remaining = one_call(text)
    assert call.name == "search"
    assert "好的" in remaining and "稍等" in remaining
    assert "DSML" not in remaining


def test_missing_close_tag_still_parses() -> None:
    text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="f">'
        '<｜DSML｜ parameter name="v" string="true">tail'
    )
    call, _ = one_call(text)
    assert call.name == "f"
    assert call.arguments["v"] == "tail"


def test_no_dsml_returns_none() -> None:
    assert parse_dsml("just a normal answer") is None
    assert looks_like_dsml("just a normal answer") is False
    assert looks_like_dsml(V41) is True
