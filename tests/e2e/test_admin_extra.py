"""Admin panel, stats, runtime logs, session listing and oversized fallback."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from conftest import API_KEY, ROOT, build_config
from fake_upstream import FakeState, make_app, start_server

from ds2api.server.app import create_app

pytestmark = pytest.mark.asyncio


async def _setup_token(client: httpx.AsyncClient, base: str) -> str:
    response = await client.post(f"{base}/admin/api/setup", json={"password": "secret1"})
    assert response.status_code == 200, response.text
    return response.json()["token"]


async def test_admin_page_served(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{stack.base_url}/admin")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "DS2Api" in response.text
    assert "setup-status" in response.text


async def test_root_redirects_to_admin(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{stack.base_url}/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/admin"


async def test_setup_status_is_public(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        before = await client.get(f"{stack.base_url}/admin/api/setup-status")
        assert before.status_code == 200
        assert before.json() == {"password_set": False}

        await _setup_token(client, stack.base_url)

        after = await client.get(f"{stack.base_url}/admin/api/setup-status")
        assert after.json() == {"password_set": True}


async def test_stats_are_recorded_after_requests(stack) -> None:
    stack.state.response_text = "hi"
    stack.state.usage = 7

    async with httpx.AsyncClient(timeout=20) as client:
        token = await _setup_token(client, stack.base_url)
        headers = {"Authorization": f"Bearer {token}"}

        for _ in range(2):
            response = await client.post(
                f"{stack.base_url}/v1/chat/completions",
                json={"model": "deepseek-default", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert response.status_code == 200

        stats = (await client.get(f"{stack.base_url}/admin/api/stats", headers=headers)).json()
        assert stats["total_requests"] == 2
        assert stats["success_requests"] == 2
        assert stats["total_completion_tokens"] == 14
        assert stats["model_stats"]["deepseek-default"]["requests"] == 2

        logs = (await client.get(f"{stack.base_url}/admin/api/logs?limit=10", headers=headers)).json()
        assert len(logs["logs"]) == 2
        assert logs["logs"][0]["model"] == "deepseek-default"
        assert logs["logs"][0]["api_key"].endswith("***")


async def test_streaming_is_recorded(stack) -> None:
    stack.state.response_text = "streamed"
    stack.state.usage = 4

    async with httpx.AsyncClient(timeout=20) as client:
        token = await _setup_token(client, stack.base_url)
        async with client.stream(
            "POST",
            f"{stack.base_url}/v1/chat/completions",
            json={
                "model": "deepseek-default",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
        ) as response:
            async for _ in response.aiter_lines():
                pass

        stats = (
            await client.get(
                f"{stack.base_url}/admin/api/stats",
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()
    assert stats["total_requests"] == 1
    assert stats["total_completion_tokens"] == 4


async def test_runtime_logs_are_captured(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        token = await _setup_token(client, stack.base_url)
        data = (
            await client.get(
                f"{stack.base_url}/admin/api/runtime-logs?limit=50",
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()
    assert data["total"] > 0
    assert data["logs"]
    entry = data["logs"][0]
    assert {"timestamp", "level", "target", "message"} <= set(entry)


async def test_sessions_endpoint(stack) -> None:
    stack.state.extra_frames = []
    async with httpx.AsyncClient(timeout=10) as client:
        token = await _setup_token(client, stack.base_url)
        response = await client.get(
            f"{stack.base_url}/admin/api/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["id"] == "sess-1"
    assert body["next_updated_at"] == 1775386361.5
    assert stack.state.by_path("/chat_session/fetch_page")


# -- oversized prompt fallback ---------------------------------------------


@pytest_asyncio.fixture
async def tiny_limit_stack(tmp_path, monkeypatch):
    state = FakeState()
    state.response_text = "ok"
    fake_server, fake_port = await start_server(make_app(state))
    config = build_config(f"http://127.0.0.1:{fake_port}/api/v0")
    # 100 chars => threshold 75, so anything longer triggers the fallback.
    config.ds_core.input_character_limits = [100]

    cache = tmp_path / "cache"
    cache.mkdir()
    wasm = ROOT / "data" / "cache" / "sha3_wasm_bg.wasm"
    if wasm.exists():
        (cache / "sha3_wasm_bg.wasm").write_bytes(wasm.read_bytes())
    monkeypatch.setenv("DS_DATA_DIR", str(tmp_path))

    app = create_app(config, tmp_path / "config.toml")
    api_server, api_port = await start_server(app)
    try:
        yield state, f"http://127.0.0.1:{api_port}"
    finally:
        api_server.should_exit = True
        fake_server.should_exit = True
        await asyncio.sleep(0.3)


async def test_oversized_prompt_uploads_history(tiny_limit_stack) -> None:
    state, base = tiny_limit_stack
    long_text = "请阅读这段很长的材料。" * 40

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "deepseek-default",
                "messages": [
                    {"role": "system", "content": "你是一个助手"},
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "收到"},
                    {"role": "user", "content": "继续"},
                ],
            },
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    assert response.status_code == 200, response.text

    uploads = state.by_path("/file/upload_file")
    assert uploads, "history should have been uploaded as a file"

    completion = state.last("/chat/completion")
    assert completion is not None
    body = completion["body"]
    assert body["ref_file_ids"], "completion should reference the uploaded file"
    # The inline tail keeps only the assistant anchor, not the whole history.
    assert len(body["prompt"]) < len(long_text)
    assert "请阅读这段很长的材料" not in body["prompt"]
