"""Shared fixtures: a fake upstream plus the real DS2Api app, both on real ports."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

# curl_cffi's asyncio backend needs add_reader, which the Windows Proactor loop
# does not implement. Selector is required on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ.setdefault("DS_DATA_DIR", str(ROOT))

from fake_upstream import FakeState, make_app, start_server  # noqa: E402

from ds2api.server.app import create_app  # noqa: E402
from ds2api.settings import Config  # noqa: E402

API_KEY = "sk-test"


def build_config(api_base: str) -> Config:
    return Config.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 0, "cors_origins": ["*"]},
            "ds_core": {
                "api_base": api_base,
                "accounts": [
                    {
                        "email": "tester@example.com",
                        "mobile": "",
                        "area_code": "",
                        "password": "pw",
                        "device_id": "device-abc",
                    }
                ],
                "hif_enabled": False,
                "startup_health_check": False,
                "model_types": ["default"],
                "max_input_tokens": [100000],
                "max_output_tokens": [8000],
                "input_character_limits": [100000],
                "hourly_request_quota": 0,
                "session_policy": "reuse",
            },
            "api_keys": [{"key": API_KEY, "description": "tests"}],
            "admin": {"password_hash": "", "jwt_secret": "0" * 64, "jwt_issued_at": 0},
        }
    )


class Stack:
    def __init__(self, state: FakeState, base_url: str, config: Config) -> None:
        self.state = state
        self.base_url = base_url
        self.config = config

    def headers(self, *, key: str | None = API_KEY) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"} if key else {}


@pytest_asyncio.fixture
async def stack(tmp_path):
    state = FakeState()
    fake_server, fake_port = await start_server(make_app(state))
    config = build_config(f"http://127.0.0.1:{fake_port}/api/v0")

    # Isolate stats.json and the wasm cache per test.
    cache = tmp_path / "cache"
    cache.mkdir()
    wasm = ROOT / "data" / "cache" / "sha3_wasm_bg.wasm"
    if wasm.exists():
        (cache / "sha3_wasm_bg.wasm").write_bytes(wasm.read_bytes())
    os.environ["DS_DATA_DIR"] = str(tmp_path)

    config_path = tmp_path / "config.toml"
    api_app = create_app(config, config_path)
    api_server, api_port = await start_server(api_app)

    yield Stack(state, f"http://127.0.0.1:{api_port}", config)

    api_server.should_exit = True
    fake_server.should_exit = True
    await asyncio.sleep(0.3)
