"""End-to-end tests: real uvicorn servers on both sides, real HTTP."""

from __future__ import annotations

import json

import httpx
import pytest

from ds2api.openai.response.tags import EOS, ROLE_USER

pytestmark = pytest.mark.asyncio


async def post_chat(stack, payload: dict, *, key: str | None = "sk-test", timeout: float = 30.0):
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(
            f"{stack.base_url}/v1/chat/completions",
            json=payload,
            headers=stack.headers(key=key),
        )


async def test_requires_api_key(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{stack.base_url}/v1/chat/completions",
            json={"model": "deepseek-default", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert response.headers.get("x-request-id", "").startswith("req-")


async def test_lowercase_bearer_rejected(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{stack.base_url}/v1/models", headers={"Authorization": "bearer sk-test"}
        )
    assert response.status_code == 401


async def test_models_endpoint(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{stack.base_url}/v1/models", headers=stack.headers())
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "deepseek-default" in ids


async def test_non_streaming(stack) -> None:
    stack.state.think_text = "思考中"
    stack.state.response_text = "你好"
    stack.state.usage = 11

    response = await post_chat(
        stack,
        {"model": "deepseek-default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "你好"
    assert body["choices"][0]["message"]["reasoning_content"] == "思考中"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 11
    assert response.headers["x-ds-account"]


async def test_upstream_request_shape(stack) -> None:
    await post_chat(
        stack,
        {
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 100,
            "stop": ["END"],
        },
    )

    completion = stack.state.last("/chat/completion")
    assert completion is not None
    body = completion["body"]

    # Fields that must never be forwarded.
    for forbidden in ("temperature", "top_p", "max_tokens", "stop", "seed", "logprobs"):
        assert forbidden not in body

    assert body["source"] == "input"
    assert body["preempt"] is False
    assert body["action"] is None
    assert body["model_type"] == "default"
    assert body["thinking_enabled"] is True
    assert body["search_enabled"] is False
    assert body["ref_file_ids"] == []
    # First turn: the key must be absent, not null.
    assert "parent_message_id" not in body

    headers = completion["headers"]
    # App identity: the x-client-* set is present, browser-only headers are not.
    for name in (
        "x-client-version",
        "x-client-platform",
        "x-client-locale",
        "x-client-bundle-id",
        "x-device-id",
        "x-client-timezone-offset",
        "x-ds-pow-response",
    ):
        assert name in headers, f"missing header {name}"
    assert headers["user-agent"].startswith("DeepSeek/")
    assert headers["x-client-platform"] == "android"

    for name in (
        "origin",
        "cookie",
        "sec-fetch-user",
        "upgrade-insecure-requests",
        "sec-ch-ua",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
    ):
        assert name not in headers, f"{name} must not be sent by the app identity"
    assert headers["authorization"] == "Bearer tok-1"


async def test_prompt_format(stack) -> None:
    await post_chat(
        stack, {"model": "deepseek-default", "messages": [{"role": "user", "content": "你好"}]}
    )
    prompt = stack.state.last("/chat/completion")["body"]["prompt"]
    assert prompt == f"{EOS}{ROLE_USER}你好<｜Assistant｜>\n"


async def test_reasoning_effort_none_disables_thinking(stack) -> None:
    await post_chat(
        stack,
        {
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
        },
    )
    assert stack.state.last("/chat/completion")["body"]["thinking_enabled"] is False


async def test_streaming(stack) -> None:
    stack.state.think_text = "想"
    stack.state.response_text = "答"
    stack.state.usage = 3

    frames: list[str] = []
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST",
            f"{stack.base_url}/v1/chat/completions",
            json={
                "model": "deepseek-default",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=stack.headers(),
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            async for line in response.aiter_lines():
                if line:
                    frames.append(line)

    assert frames[-1] == "data: [DONE]"
    assert sum(1 for f in frames if f == "data: [DONE]") == 1

    chunks = [json.loads(f[len("data: "):]) for f in frames if f != "data: [DONE]"]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[0]["usage"]["completion_tokens"] == 0

    reasoning_index = next(
        i for i, c in enumerate(chunks) if c["choices"][0]["delta"].get("reasoning_content")
    )
    content_index = next(
        i for i, c in enumerate(chunks) if c["choices"][0]["delta"].get("content")
    )
    assert reasoning_index < content_index

    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] == 3


async def test_dsml_tool_call_end_to_end(stack) -> None:
    stack.state.response_text = (
        '<｜DSML｜ calls><｜DSML｜ invoke name="get_weather">'
        '<｜DSML｜ parameter name="city" string="true">上海</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>'
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    )
    response = await post_chat(
        stack,
        {
            "model": "deepseek-default",
            "messages": [{"role": "user", "content": "天气"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] == ""
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "上海", "days": 3}

    # The tool spec must have been injected into the system prompt.
    prompt = stack.state.last("/chat/completion")["body"]["prompt"]
    assert "## Tools" in prompt
    assert "get_weather" in prompt


async def test_session_reuse_second_turn(stack) -> None:
    first = await post_chat(
        stack, {"model": "deepseek-default", "messages": [{"role": "user", "content": "第一问"}]}
    )
    assert first.status_code == 200

    second = await post_chat(
        stack,
        {
            "model": "deepseek-default",
            "messages": [
                {"role": "user", "content": "第一问"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "第二问"},
            ],
        },
    )
    assert second.status_code == 200

    completions = stack.state.by_path("/chat/completion")
    assert len(completions) == 2
    first_body, second_body = completions[0]["body"], completions[1]["body"]

    assert first_body["chat_session_id"] == second_body["chat_session_id"]
    assert second_body["parent_message_id"] == 2
    # Reuse sends only the newest user message, with no ChatML tags.
    assert second_body["prompt"] == "第二问"


async def test_rate_limit_maps_to_429(stack) -> None:
    stack.state.rate_limit = True
    response = await post_chat(
        stack, {"model": "deepseek-default", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "30"
    assert response.json()["error"]["code"] == "overloaded"


async def test_empty_messages_rejected(stack) -> None:
    response = await post_chat(stack, {"model": "deepseek-default", "messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


async def test_n_greater_than_one_rejected(stack) -> None:
    response = await post_chat(
        stack,
        {"model": "deepseek-default", "messages": [{"role": "user", "content": "hi"}], "n": 2},
    )
    assert response.status_code == 400


async def test_health(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{stack.base_url}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
