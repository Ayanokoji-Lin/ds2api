"""HIF risk-control tokens.

The official web client polls two unauthenticated endpoints and echoes the value
back in ``x-hif-leim`` / ``x-hif-dliq`` on the SSE request. Missing these is the
single most common reason a third-party gateway gets flagged, so they are on by
default.

Behaviour copied from ``ds_core/src/accounts/hif.rs``:

* TTL comes from the ``x-hif-ttl`` response header (default 600s)
* refresh 30s early, never below 1s
* failures back off 30s, 60s, ... capped at 600s
* cached per ``X-Device-Id`` (one device identity per account)
* if a token cannot be fetched, the header is simply omitted
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .identity import ClientIdentity

logger = logging.getLogger("ds2api.hif")

LEIM_URL = "https://hif-leim.deepseek.com/query"
DLIQ_URL = "https://hif-dliq.deepseek.com/query"

DEFAULT_TTL_SECS = 600
REFRESH_MARGIN = 30
INITIAL_FAILURE_BACKOFF = 30
MAX_FAILURE_BACKOFF = 600
FETCH_TIMEOUT = 3.0

_HEADER_NAMES = {LEIM_URL: "X-Hif-Leim", DLIQ_URL: "X-Hif-Dliq"}


@dataclass(slots=True)
class _Entry:
    value: str = ""
    expire_at: float = 0.0
    failures: int = 0
    next_attempt: float = 0.0

    def valid(self, now: float) -> bool:
        return bool(self.value) and now < self.expire_at

    def backoff(self, now: float) -> None:
        self.failures += 1
        delay = min(INITIAL_FAILURE_BACKOFF * (2 ** (self.failures - 1)), MAX_FAILURE_BACKOFF)
        self.next_attempt = now + delay


class HifManager:
    def __init__(self, http, *, enabled: bool = True) -> None:
        self._http = http
        self.enabled = enabled
        # device_id -> url -> entry
        self._cache: dict[str, dict[str, _Entry]] = {}

    def _entries(self, device_id: str) -> dict[str, _Entry]:
        return self._cache.setdefault(device_id, {LEIM_URL: _Entry(), DLIQ_URL: _Entry()})

    async def headers_for(self, identity: ClientIdentity) -> dict[str, str]:
        if not self.enabled:
            return {}
        entries = self._entries(identity.device_id)
        now = time.monotonic()
        out: dict[str, str] = {}
        for url, entry in entries.items():
            if not entry.valid(now) and now >= entry.next_attempt:
                await self._refresh(identity, url, entry)
            if entry.valid(time.monotonic()):
                out[_HEADER_NAMES[url]] = entry.value
        return out

    async def _refresh(self, identity: ClientIdentity, url: str, entry: _Entry) -> None:
        try:
            session = self._http.session
            response = await session.get(
                url,
                headers=identity.headers(site="cross-site"),
                timeout=FETCH_TIMEOUT,
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise ValueError(f"HTTP {response.status_code}")
            payload = response.json()
            value = (
                payload.get("data", {})
                .get("biz_data", {})
                .get("value")
                if isinstance(payload, dict)
                else None
            )
            if not isinstance(value, str) or not value:
                raise ValueError("missing data.biz_data.value")
            ttl_raw = response.headers.get("x-hif-ttl")
            ttl = int(ttl_raw) if ttl_raw and str(ttl_raw).isdigit() else DEFAULT_TTL_SECS
            ttl = max(1, ttl)
            now = time.monotonic()
            entry.value = value
            entry.expire_at = now + max(1, ttl - REFRESH_MARGIN)
            entry.failures = 0
            entry.next_attempt = 0.0
            logger.debug("hif token refreshed from %s (ttl=%ss)", url, ttl)
        except Exception as exc:  # noqa: BLE001 - never block the request
            entry.backoff(time.monotonic())
            logger.warning("hif fetch failed (%s): %s", url, exc)
