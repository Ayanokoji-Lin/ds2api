"""OpenAI error envelope.

Only one flavour exists now that Anthropic compatibility was dropped:

    {"error": {"message": ..., "type": ..., "param": null, "code": ...}}
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from ..core.errors import (
    BadRequest,
    CoreError,
    Overloaded,
    ProofOfWorkFailed,
    ProviderError,
    StatusError,
    StreamError,
)


class ApiError(Exception):
    def __init__(self, status: int, message: str, error_type: str, code: str) -> None:
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(message)

    def response(self, *, request_id: str | None = None) -> JSONResponse:
        headers = {}
        if self.status == 429:
            headers["Retry-After"] = "30"
        if request_id:
            headers["x-request-id"] = request_id
        return JSONResponse(
            status_code=self.status,
            content={
                "error": {
                    "message": self.message,
                    "type": self.error_type,
                    "param": None,
                    "code": self.code,
                }
            },
            headers=headers,
        )


def unauthorized(message: str = "invalid api token") -> ApiError:
    return ApiError(401, message, "invalid_request_error", "invalid_api_key")


def not_found(message: str) -> ApiError:
    return ApiError(404, message, "invalid_request_error", "model_not_found")


def from_core(exc: CoreError) -> ApiError:
    if isinstance(exc, BadRequest):
        return ApiError(400, str(exc), "invalid_request_error", "bad_request")
    if isinstance(exc, Overloaded):
        return ApiError(429, str(exc), "server_error", "overloaded")
    if isinstance(exc, StatusError):
        return ApiError(502, str(exc), "server_error", "provider_error")
    if isinstance(exc, (ProviderError, StreamError)):
        return ApiError(502, str(exc), "server_error", "provider_error")
    if isinstance(exc, ProofOfWorkFailed):
        return ApiError(500, f"proof of work failed: {exc}", "server_error", "internal_error")
    return ApiError(500, str(exc), "server_error", "internal_error")
