"""Identity construction, including the web -> app downgrade."""

from __future__ import annotations

from ds2api.core.identity import ClientIdentity, account_device_uuid, derive_device_uuid
from ds2api.core.pool import _build_identity
from ds2api.settings import AccountConfig, Config


def make_config(**core_overrides) -> Config:
    core = {
        "accounts": [{"email": "a@b.c", "password": "p", "device_id": "fp"}],
        "model_types": ["default"],
        "max_input_tokens": [1000],
        "max_output_tokens": [1000],
        "input_character_limits": [10000],
    }
    core.update(core_overrides)
    return Config.model_validate({"ds_core": core})


def test_default_identity_is_app() -> None:
    config = make_config()
    identity = _build_identity(config, config.ds_core.accounts[0])
    assert identity.client_os == "android"
    assert identity.client_platform == "android"
    assert identity.user_agent.startswith("DeepSeek/")
    assert identity.is_app


def test_web_identity_downgrades_to_app() -> None:
    config = make_config(
        client_os="web",
        client_platform="web",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0.0.0",
        client_device_model="",
    )
    identity = _build_identity(config, config.ds_core.accounts[0])
    assert identity.client_os == "android"
    assert identity.client_platform == "android"
    assert identity.user_agent == "DeepSeek/2.5.0 Android/35"
    assert identity.device_model == "Pixel 7"


def test_ios_is_preserved() -> None:
    config = make_config(client_os="ios", client_platform="ios")
    identity = _build_identity(config, config.ds_core.accounts[0])
    assert identity.client_os == "ios"
    assert identity.is_app


def test_app_headers_omit_browser_fingerprint() -> None:
    identity = ClientIdentity(device_id="d", client_platform="android", client_os="android")
    headers = identity.headers()
    assert headers["Accept"] == "*/*"
    assert headers["X-Client-Platform"] == "android"
    for name in ("Sec-Ch-Ua", "Sec-Fetch-Dest", "Sec-Fetch-Mode", "Sec-Fetch-Site"):
        assert headers[name] is None, f"{name} must be removed for the app identity"


def test_web_headers_keep_browser_fingerprint() -> None:
    identity = ClientIdentity(
        device_id="d",
        client_platform="web",
        client_os="web",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0.0.0",
    )
    headers = identity.headers()
    assert headers["Sec-Ch-Ua"].startswith('"Chromium"')
    assert headers["Sec-Fetch-Dest"] == "empty"
    assert headers["Sec-Fetch-User"] is None


def test_device_uuid_is_stable_and_v4_shaped() -> None:
    first = derive_device_uuid(b"seed")
    assert first == derive_device_uuid(b"seed")
    assert first != derive_device_uuid(b"other")
    assert first[14] == "4"
    assert first[19] in "89ab"
    assert len(first) == 36


def test_account_device_uuid_differs_per_fingerprint() -> None:
    assert account_device_uuid("fp-1") != account_device_uuid("fp-2")
    assert account_device_uuid(" fp-1 ") == account_device_uuid("fp-1")
