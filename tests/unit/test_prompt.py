"""Prompt construction tests (byte-exact expectations)."""

from __future__ import annotations

from ds2api.openai.request.prompt import build, merge_messages, role_tag
from ds2api.openai.request.tools import extract
from ds2api.openai.response.tags import EOS, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER
from ds2api.openai.types import ChatCompletionsRequest, Message


def req(**kwargs) -> ChatCompletionsRequest:
    return ChatCompletionsRequest(**kwargs)


def test_single_user_message() -> None:
    prompt = build(req(messages=[Message(role="user", content="hi")]), extract(req()))
    assert prompt == f"{EOS}{ROLE_USER}hi{ROLE_ASSISTANT}\n"


def test_system_message_kept_first() -> None:
    prompt = build(
        req(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="user", content="hi"),
            ]
        ),
        extract(req()),
    )
    assert prompt == f"{ROLE_SYSTEM}be brief{EOS}{ROLE_USER}hi{ROLE_ASSISTANT}\n"


def test_consecutive_same_role_merged() -> None:
    merged = merge_messages(
        [
            Message(role="user", content="a"),
            Message(role="user", content="b"),
            Message(role="assistant", content="c"),
        ]
    )
    assert len(merged) == 2
    assert merged[0].content == "a\nb"


def test_tool_messages_are_not_merged_by_merge_messages() -> None:
    merged = merge_messages(
        [
            Message(role="tool", content="r1", tool_call_id="1"),
            Message(role="tool", content="r2", tool_call_id="2"),
        ]
    )
    assert len(merged) == 2


def test_dsml_tool_results_ride_in_user_turn() -> None:
    prompt = build(
        req(
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="", tool_calls=[
                    {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]),
                Message(role="tool", content="r1", tool_call_id="1"),
                Message(role="tool", content="r2", tool_call_id="2"),
            ]
        ),
        extract(req(), protocol="dsml"),
    )
    assert f"{EOS}{ROLE_USER}<tool_result>r1</tool_result><tool_result>r2</tool_result>" in prompt
    # tool results come before the final assistant anchor
    assert prompt.endswith(f"{ROLE_ASSISTANT}\n")


def test_legacy_tool_results_use_tool_outputs_tags() -> None:
    prompt = build(
        req(
            messages=[
                Message(role="user", content="q"),
                Message(role="tool", content="r1", tool_call_id="1"),
            ]
        ),
        extract(req(), protocol="legacy"),
    )
    assert "<｜tool▁outputs▁begin｜>" in prompt
    assert "<tool_result>" not in prompt


def test_assistant_tool_calls_rendered_as_legacy_block() -> None:
    prompt = build(
        req(
            messages=[
                Message(role="user", content="q"),
                Message(
                    role="assistant",
                    content="thinking",
                    tool_calls=[
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1}'},
                        }
                    ],
                ),
            ]
        ),
        extract(req()),
    )
    assert "<|tool▁calls▁begin|>" in prompt
    assert '{"name": "f", "arguments": {"a": 1}}' in prompt


def test_prefill_assistant_not_duplicated() -> None:
    prompt = build(
        req(
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="partial"),
            ]
        ),
        extract(req()),
    )
    assert prompt.endswith("partial")
    assert prompt.count(ROLE_ASSISTANT) == 1


def test_tools_injected_into_existing_system() -> None:
    request = req(
        messages=[
            Message(role="system", content="base rules"),
            Message(role="user", content="hi"),
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
    )
    prompt = build(request, extract(request))
    assert prompt.startswith(ROLE_SYSTEM)
    assert "base rules" in prompt
    assert "## Tools" in prompt
    assert "get_weather" in prompt
    # the injected block sits before the first user turn
    assert prompt.index("## Tools") < prompt.index(ROLE_USER)


def test_tool_choice_none_skips_injection() -> None:
    request = req(
        messages=[Message(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="none",
    )
    prompt = build(request, extract(request))
    assert "## Tools" not in prompt


def test_response_format_json_schema() -> None:
    request = req(
        messages=[Message(role="user", content="hi")],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
    )
    prompt = build(request, extract(request))
    assert "~~~json" in prompt


def test_role_tag_capitalises() -> None:
    assert role_tag("user") == "<｜User｜>"
    assert role_tag("assistant") == "<｜Assistant｜>"
