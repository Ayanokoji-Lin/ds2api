"""Tests for the ChatML block parser and the oversized-prompt splitter."""

from __future__ import annotations

from ds2api.core.chat import parse_native_blocks, split_history_prompt
from ds2api.openai.request.prompt import build
from ds2api.openai.request.tools import extract
from ds2api.openai.types import ChatCompletionsRequest, Message


def make_prompt(*messages: Message) -> str:
    request = ChatCompletionsRequest(messages=list(messages))
    return build(request, extract(request))


def test_parse_blocks() -> None:
    prompt = "<｜System｜>sys\n<｜User｜>hi<｜Assistant｜>\nanswer\n"
    blocks = parse_native_blocks(prompt)
    assert blocks[0] == ("system", "sys")
    assert ("user", "hi") in blocks
    assert blocks[-1][0] == "assistant"


def test_split_keeps_last_assistant_inline() -> None:
    prompt = make_prompt(
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="second"),
    )
    inline, history = split_history_prompt(prompt)
    assert inline == "<｜Assistant｜>\n"
    assert "[file content end]" in history
    assert "[file name]: IGNORE" in history
    assert "[file content begin]" in history
    # Everything before the trailing anchor goes into the file - including the
    # newest user turn, which is how the Rust implementation behaves.
    assert "first" in history
    assert "reply" in history
    assert "second" in history


def test_split_preserves_marker_tag_casing() -> None:
    """The Rust parser rewrites <|end▁of▁sentence|> to <|End▁of▁sentence|>; we do not."""
    prompt = make_prompt(
        Message(role="user", content="a"),
        Message(role="assistant", content="b"),
        Message(role="user", content="c"),
    )
    _, history = split_history_prompt(prompt)
    assert "end▁of▁sentence" in history
    assert "End▁of▁sentence" not in history


def test_split_without_assistant_returns_whole_prompt() -> None:
    inline, history = split_history_prompt("<｜User｜>only\n")
    assert inline == "<｜User｜>only\n"
    assert history == ""


def test_split_history_contains_every_prior_block() -> None:
    prompt = make_prompt(
        Message(role="system", content="rules"),
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
    )
    inline, history = split_history_prompt(prompt)
    assert "rules" in history and "q1" in history and "a1" in history and "q2" in history
    # Only the generation anchor stays inline.
    assert inline == "<｜Assistant｜>\n"
