"""Admin API end-to-end tests."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def test_admin_requires_setup_then_jwt(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        # No password yet: login is refused, setup works.
        login = await client.post(f"{stack.base_url}/admin/api/login", json={"password": "secret1"})
        assert login.status_code == 403

        setup = await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret1"})
        assert setup.status_code == 200
        token = setup.json()["token"]

        # Second setup is refused.
        again = await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret2"})
        assert again.status_code == 403

        # Protected endpoint needs the token.
        unauth = await client.get(f"{stack.base_url}/admin/api/status")
        assert unauth.status_code == 401

        authed = await client.get(
            f"{stack.base_url}/admin/api/status", headers={"Authorization": f"Bearer {token}"}
        )
        assert authed.status_code == 200
        assert authed.json()["accounts"] == 1


async def test_admin_login_wrong_password(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret1"})
        bad = await client.post(f"{stack.base_url}/admin/api/login", json={"password": "nope"})
        assert bad.status_code == 401

        good = await client.post(f"{stack.base_url}/admin/api/login", json={"password": "secret1"})
        assert good.status_code == 200
        assert good.json()["token"]


async def test_short_password_rejected(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "123"})
    assert response.status_code == 400


async def test_config_roundtrip_keeps_secrets(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        token = (
            await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret1"})
        ).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        current = (await client.get(f"{stack.base_url}/admin/api/config", headers=headers)).json()
        # Password hash is never echoed.
        assert "password_hash" not in current["admin"]
        assert current["admin"]["password_set"] is True
        assert current["ds_core"]["accounts"][0]["password"] == "pw"

        # Simulate the dashboard posting the config back with masked passwords.
        current["ds_core"]["accounts"][0]["password"] = ""
        current["ds_core"]["hourly_request_quota"] = 42
        response = await client.put(
            f"{stack.base_url}/admin/api/config", json=current, headers=headers
        )
        assert response.status_code == 200, response.text
        updated = response.json()
        assert updated["ds_core"]["hourly_request_quota"] == 42
        # The existing password survived the round trip.
        assert updated["ds_core"]["accounts"][0]["password"] == "pw"


async def test_config_rejects_misaligned_limits(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        token = (
            await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret1"})
        ).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        current = (await client.get(f"{stack.base_url}/admin/api/config", headers=headers)).json()
        current["ds_core"]["max_input_tokens"] = [1, 2, 3]
        response = await client.put(
            f"{stack.base_url}/admin/api/config", json=current, headers=headers
        )
        assert response.status_code == 400


async def test_stats_and_logs_endpoints(stack) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        token = (
            await client.post(f"{stack.base_url}/admin/api/setup", json={"password": "secret1"})
        ).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        stats = await client.get(f"{stack.base_url}/admin/api/stats", headers=headers)
        assert stats.status_code == 200
        assert "total_requests" in stats.json()

        logs = await client.get(f"{stack.base_url}/admin/api/logs?limit=5", headers=headers)
        assert logs.status_code == 200
        assert "logs" in logs.json()

        models = await client.get(f"{stack.base_url}/admin/api/models", headers=headers)
        assert models.status_code == 200
        assert models.json()["data"]
