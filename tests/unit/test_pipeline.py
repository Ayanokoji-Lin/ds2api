"""Streaming pipeline tests against synthetic event streams."""

from __future__ import annotations

import json

import pytest

from ds2api.core.events import ContentDelta, Done, Meta, ThinkDelta
from ds2api.openai.response.pipeline import aggregate, iter_chunks, stream_sse

DSML = (
    '<｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
    '<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>'
    "</｜DSML｜ invoke></｜DSML｜ calls>"
)


async def events(*items):
    for item in items:
        yield item


async def collect_sse(items, **kwargs):
    raw = []
    async for chunk in stream_sse(events(*items), model="deepseek-default", prompt_tokens=7, **kwargs):
        raw.append(chunk)
    return raw


async def collect_chunks(items, **kwargs):
    out = []
    async for chunk in iter_chunks(
        events(*items), model="deepseek-default", prompt_tokens=7, include_obfuscation=False, **kwargs
    ):
        out.append(chunk)
    return out


def parse_frames(raw: list[bytes]) -> list[dict | str]:
    frames: list[dict | str] = []
    for block in raw:
        text = block.decode()
        assert text.startswith("data: ")
        body = text[len("data: ") :].rstrip("\n")
        frames.append(body if body == "[DONE]" else json.loads(body))
    return frames


@pytest.mark.asyncio
async def test_basic_stream_shape() -> None:
    raw = await collect_sse([Meta("acct"), ThinkDelta("think"), ContentDelta("hi"), Done("stop", 5)])
    frames = parse_frames(raw)

    assert frames[-1] == "[DONE]"
    assert sum(1 for f in frames if f == "[DONE]") == 1

    first = frames[0]
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["usage"] == {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7}

    assert frames[1]["choices"][0]["delta"]["reasoning_content"] == "think"
    assert frames[2]["choices"][0]["delta"]["content"] == "hi"

    last = frames[-2]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"] == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}


@pytest.mark.asyncio
async def test_every_chunk_has_distinct_id() -> None:
    frames = parse_frames(await collect_sse([Meta("a"), ContentDelta("x"), Done("stop", 1)]))
    ids = [f["id"] for f in frames if isinstance(f, dict)]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_dsml_split_into_single_characters() -> None:
    items = [Meta("a"), *(ContentDelta(ch) for ch in DSML), Done("stop", 9)]
    frames = parse_frames(await collect_sse(items))

    calls = []
    content = ""
    finish = None
    for frame in frames:
        if frame == "[DONE]":
            continue
        delta = frame["choices"][0]["delta"]
        content += delta.get("content") or ""
        if delta.get("tool_calls"):
            calls.extend(delta["tool_calls"])
        if frame["choices"][0].get("finish_reason"):
            finish = frame["choices"][0]["finish_reason"]

    assert finish == "tool_calls"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "上海"}
    assert "DSML" not in content


@pytest.mark.asyncio
async def test_prose_before_dsml_is_preserved() -> None:
    items = [Meta("a"), ContentDelta("我来查一下。"), *(ContentDelta(ch) for ch in DSML), Done("stop", 9)]
    frames = parse_frames(await collect_sse(items))
    content = "".join(
        f["choices"][0]["delta"].get("content") or "" for f in frames if isinstance(f, dict)
    )
    assert "我来查一下。" in content


@pytest.mark.asyncio
async def test_stop_sequence_truncates_across_chunks() -> None:
    items = [Meta("a"), ContentDelta("hello wor"), ContentDelta("ld STOP tail"), Done("stop", 3)]
    frames = parse_frames(await collect_sse(items, stop=["STOP"]))
    content = "".join(
        f["choices"][0]["delta"].get("content") or "" for f in frames if isinstance(f, dict)
    )
    assert content == "hello world "
    assert "tail" not in content


@pytest.mark.asyncio
async def test_aggregate_matches_stream() -> None:
    result = await aggregate(
        events(Meta("a"), ThinkDelta("why"), ContentDelta("answer"), Done("stop", 4)),
        model="deepseek-default",
        prompt_tokens=3,
    )
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "answer"
    assert result["choices"][0]["message"]["reasoning_content"] == "why"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 4


@pytest.mark.asyncio
async def test_aggregate_tool_calls_content_is_empty_string() -> None:
    result = await aggregate(
        events(Meta("a"), *(ContentDelta(ch) for ch in DSML), Done("stop", 4)),
        model="deepseek-default",
        prompt_tokens=3,
    )
    message = result["choices"][0]["message"]
    assert message["content"] == ""
    assert len(message["tool_calls"]) == 1
    assert result["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_truncated_stream_flushes_tail_as_content() -> None:
    # No Done event: the generator ends, buffered text must still surface.
    frames = parse_frames(await collect_sse([Meta("a"), ContentDelta("partial answer")]))
    content = "".join(
        f["choices"][0]["delta"].get("content") or "" for f in frames if isinstance(f, dict)
    )
    assert content == "partial answer"
    assert frames[-1] == "[DONE]"
