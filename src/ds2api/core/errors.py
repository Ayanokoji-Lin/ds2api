"""Error taxonomy, mirroring ``ds_core``'s CoreError / ClientError / PowError.

The HTTP layer maps these onto OpenAI error envelopes:

    BadRequest        -> 400 invalid_request_error / bad_request
    Overloaded        -> 429 server_error / overloaded      (+ Retry-After: 30)
    ProviderError     -> 502 server_error / provider_error
    ProofOfWorkFailed -> 500 server_error / internal_error
    Internal          -> 500 server_error / internal_error
"""

from __future__ import annotations


class CoreError(Exception):
    """Base class for every upstream failure."""


class BadRequest(CoreError):
    """The caller sent something we cannot translate."""


class Overloaded(CoreError):
    """Account pool exhausted or upstream rate limit reached."""

    def __init__(self, message: str = "no account available") -> None:
        super().__init__(message)


class ProviderError(CoreError):
    """Upstream returned a business error or an unusable stream."""


class ProofOfWorkFailed(CoreError):
    """PoW challenge could not be solved."""


class StreamError(CoreError):
    """The upstream SSE stream broke mid-flight."""


class ClientError(CoreError):
    """Transport-level failure."""


class StatusError(ClientError):
    """Non-2xx HTTP response from upstream."""

    def __init__(self, status: int, body: bytes | str, *, waf: bool = False) -> None:
        self.status = status
        self.body = body if isinstance(body, str) else body.decode("utf-8", "replace")
        self.waf = waf
        detail = f"HTTP {status}"
        if waf:
            detail += " (AWS WAF challenge - try a non-US proxy via [proxy].url)"
        super().__init__(f"{detail}: {self.body[:400]}")


class BusinessError(ClientError):
    """Upstream envelope carried a non-zero code / biz_code."""

    def __init__(self, code: int, message: str, *, biz: bool = True) -> None:
        self.code = code
        self.message = message
        self.biz = biz
        super().__init__(f"Business error: code={code}, msg={message}")


# Login failures that must not be retried: retrying a muted/banned account only
# extends the penalty. Mirrors ``is_terminal_login_error`` in pool.rs.
TERMINAL_BIZ_CODES: frozenset[int] = frozenset({2, 5, 10, 11})
