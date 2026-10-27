"""HTTP transport against the DeepSeek web backend.

Uses curl_cffi so the TLS/JA3 and HTTP/2 fingerprint matches a real Chrome. Plain
httpx/requests are not viable here: the upstream sits behind CloudFront + WAF and
fingerprints the client.

Two WAF behaviours are handled explicitly:

* ``202`` with an ``x-amzn-waf-action`` header means the request was challenged
  (usually because the egress IP is in a blocked region). We surface it with an
  actionable message instead of retrying.
* Any other non-2xx becomes :class:`StatusError`.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from .errors import BusinessError, StatusError

logger = logging.getLogger("ds2api.http")

PROFILE_CANDIDATES = ("chrome136", "chrome133a", "chrome131", "chrome124", "chrome120", "chrome")

#: curl's enum value for HTTP/1.1 (``CurlHttpVersion.V1_1``). The libcurl binding
#: wants the integer, not 1.1.
HTTP_VERSION_1_1 = 2


def pick_profile(preferred: str | None = None) -> str:
    """Return the first impersonation profile curl_cffi accepts.

    Uses the *synchronous* Session for probing: on curl_cffi >= 0.15
    ``AsyncSession.close`` is a coroutine, so probing with it leaks.
    """
    from curl_cffi.requests import Session

    candidates = [preferred] if preferred else []
    candidates.extend(p for p in PROFILE_CANDIDATES if p != preferred)
    for name in candidates:
        if not name:
            continue
        try:
            Session(impersonate=name).close()
            if name != preferred:
                logger.warning("impersonate profile %r unavailable, using %r", preferred, name)
            return name
        except Exception:  # noqa: BLE001 - curl_cffi raises plain ValueError
            continue
    raise RuntimeError("no usable curl_cffi impersonation profile")


def unwrap_envelope(payload: Any) -> Any:
    """Unwrap the ``{code, msg, data:{biz_code, biz_msg, biz_data}}`` envelope."""
    if not isinstance(payload, dict):
        return payload

    code = payload.get("code", 0)
    if isinstance(code, int) and code != 0:
        raise BusinessError(code, str(payload.get("msg", "")), biz=False)

    data = payload.get("data")
    if data is None:
        raise BusinessError(-1, "response missing data", biz=False)
    if not isinstance(data, dict):
        return data

    biz_code = data.get("biz_code", 0)
    if isinstance(biz_code, int) and biz_code != 0:
        raise BusinessError(biz_code, str(data.get("biz_msg", "")), biz=True)

    return data.get("biz_data")


class UpstreamHttp:
    def __init__(
        self,
        *,
        proxy: str | None = None,
        impersonate: str | None = None,
        http_version: float | None = None,
    ) -> None:
        self._proxy = proxy
        self._profile = pick_profile(impersonate)
        # Plaintext targets need HTTP/1.1: curl's h2c upgrade confuses simple
        # servers and drops the body. Real deployments use https, where the
        # browser's HTTP/2 fingerprint should be kept.
        self._http_version = http_version
        self._session = None

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def session(self):
        """Raw curl_cffi session, for callers that need response headers."""
        return self._ensure_session()

    def _ensure_session(self):
        if self._session is None:
            from curl_cffi.requests import AsyncSession

            extra = {}
            if self._http_version is not None:
                extra["http_version"] = self._http_version
            self._session = AsyncSession(
                impersonate=self._profile, proxy=self._proxy, **extra
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            session, self._session = self._session, None
            result = session.close()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _check(response) -> None:
        status = response.status_code
        if status == 202 and "x-amzn-waf-action" in {k.lower() for k in response.headers}:
            raise StatusError(202, response.content, waf=True)
        if status < 200 or status >= 300:
            raise StatusError(status, response.content)

    async def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str | None],
        body: Any,
        timeout: float = 60.0,
    ) -> Any:
        session = self._ensure_session()
        response = await session.post(url, headers=headers, json=body, timeout=timeout)
        self._check(response)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            raise StatusError(response.status_code, response.content) from None
        return unwrap_envelope(payload)

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str | None],
        params: list[tuple[str, str]] | None = None,
        timeout: float = 60.0,
    ) -> Any:
        session = self._ensure_session()
        response = await session.get(url, headers=headers, params=params, timeout=timeout)
        self._check(response)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            raise StatusError(response.status_code, response.content) from None
        return unwrap_envelope(payload)

    async def post_multipart(
        self,
        url: str,
        *,
        headers: dict[str, str | None],
        filename: str,
        content_type: str,
        data: bytes,
        timeout: float = 120.0,
    ) -> Any:
        session = self._ensure_session()
        files = {"file": (filename, data, content_type)}
        response = await session.post(url, headers=headers, files=files, timeout=timeout)
        self._check(response)
        payload = response.json()
        return unwrap_envelope(payload)

    async def post_stream(
        self,
        url: str,
        *,
        headers: dict[str, str | None],
        body: Any,
        timeout: float | None = None,
    ):
        """Return an open streaming response. Caller must ``aclose()`` it."""
        session = self._ensure_session()
        response = await session.post(url, headers=headers, json=body, stream=True, timeout=timeout)
        self._check(response)
        return response
