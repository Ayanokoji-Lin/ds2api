"""Endpoint wrappers: one method per upstream path.

Paths and payload shapes are copied from ``ds_core/src/accounts/client.rs``.
Payloads intentionally omit ``None`` fields where the Rust struct uses
``skip_serializing_if`` - notably ``parent_message_id`` on the first turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .errors import ProviderError
from .http import UpstreamHttp
from .identity import ClientIdentity
from .pow import Challenge

logger = logging.getLogger("ds2api.endpoints")

P_USERS_LOGIN = "/users/login"
P_USERS_CURRENT = "/users/current"
P_CHECK_DEVICE = "/users/auth_token/check_device"
P_SESSION_CREATE = "/chat_session/create"
P_SESSION_DELETE = "/chat_session/delete"
P_SESSION_FETCH_PAGE = "/chat_session/fetch_page"
P_POW_CHALLENGE = "/chat/create_pow_challenge"
P_COMPLETION = "/chat/completion"
P_STOP_STREAM = "/chat/stop_stream"
P_FILE_UPLOAD = "/file/upload_file"
P_FILE_FETCH = "/file/fetch_files"

COMPLETION_TARGET = "/api/v0/chat/completion"
UPLOAD_TARGET = "/api/v0/file/upload_file"


@dataclass(slots=True)
class UserInfo:
    id: str
    token: str
    email: str | None = None
    mobile: str | None = None


@dataclass(slots=True)
class SessionInfo:
    id: str


@dataclass(slots=True)
class UploadInfo:
    id: str
    status: str = ""
    file_name: str = ""
    file_size: int = 0


@dataclass(slots=True)
class FileStatus:
    id: str
    status: str
    file_name: str = ""
    file_size: int = 0


class Endpoints:
    def __init__(self, http: UpstreamHttp, api_base: str) -> None:
        self.http = http
        self.api_base = api_base.rstrip("/")

    def url(self, path: str) -> str:
        return f"{self.api_base}{path}"

    # -- auth --------------------------------------------------------------

    async def login(self, identity: ClientIdentity, *, email: str, mobile: str,
                    password: str, area_code: str, device_id: str, os_name: str) -> UserInfo:
        # The real client always sends all six keys; the unused one is an empty
        # string, not null.
        body = {
            "email": email,
            "mobile": mobile,
            "password": password,
            "area_code": area_code,
            "device_id": device_id,
            "os": os_name,
        }
        data = await self.http.post_json(
            self.url(P_USERS_LOGIN), headers=identity.headers(login=True), body=body
        )
        # unwrap_envelope already returned biz_data
        user = data.get("user") if isinstance(data, dict) else None
        if not user:
            raise ProviderError("login response missing user")
        token = user.get("token")
        if not token:
            raise ProviderError("login response missing token")
        return UserInfo(
            id=str(user.get("id", "")),
            token=str(token),
            email=user.get("email"),
            mobile=user.get("mobile_number"),
        )

    async def check_device(self, identity: ClientIdentity, token: str) -> str | None:
        body = {"device_id": identity.device_id, "device_model": identity.device_model}
        data = await self.http.post_json(
            self.url(P_CHECK_DEVICE),
            headers=identity.headers(token=token),
            body=body,
        )
        rotate = data.get("rotate") if isinstance(data, dict) else None
        if isinstance(rotate, str) and rotate:
            return rotate
        if isinstance(rotate, dict):
            candidate = rotate.get("token")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    # -- sessions ----------------------------------------------------------

    async def create_session(self, identity: ClientIdentity, token: str) -> str:
        data = await self.http.post_json(
            self.url(P_SESSION_CREATE), headers=identity.headers(token=token), body={}
        )
        session = data.get("chat_session") if isinstance(data, dict) else None
        if not session or not session.get("id"):
            raise ProviderError("create_session response missing id")
        return str(session["id"])

    async def delete_session(self, identity: ClientIdentity, token: str, session_id: str) -> None:
        # biz_data is null here; unwrap_envelope already tolerates that.
        await self.http.post_json(
            self.url(P_SESSION_DELETE),
            headers=identity.headers(token=token),
            body={"chat_session_id": session_id},
        )

    async def fetch_sessions(self, identity: ClientIdentity, token: str,
                             updated_at: float | None = None) -> Any:
        params = [("lte_cursor.pinned", "false")]
        if updated_at is not None:
            params.append(("lte_cursor.updated_at", str(updated_at)))
        return await self.http.get_json(
            self.url(P_SESSION_FETCH_PAGE), headers=identity.headers(token=token), params=params
        )

    # -- pow ---------------------------------------------------------------

    async def create_pow_challenge(self, identity: ClientIdentity, token: str,
                                   target_path: str) -> Challenge:
        data = await self.http.post_json(
            self.url(P_POW_CHALLENGE),
            headers=identity.headers(token=token),
            body={"target_path": target_path},
        )
        challenge = data.get("challenge") if isinstance(data, dict) else None
        if not challenge:
            raise ProviderError("pow challenge missing")
        return Challenge.from_dict(challenge)

    # -- completion --------------------------------------------------------

    async def completion(
        self,
        identity: ClientIdentity,
        token: str,
        body: dict,
        *,
        pow_header: str,
        hif_headers: dict[str, str] | None = None,
    ):
        headers = identity.headers(
            token=token, pow_header=pow_header, hif=hif_headers or {}
        )
        return await self.http.post_stream(
            self.url(P_COMPLETION), headers=headers, body=body, timeout=None
        )

    async def stop_stream(self, identity: ClientIdentity, token: str,
                          session_id: str, message_id: int) -> None:
        await self.http.post_json(
            self.url(P_STOP_STREAM),
            headers=identity.headers(token=token),
            body={"chat_session_id": session_id, "message_id": message_id},
        )

    # -- files -------------------------------------------------------------

    async def upload_file(
        self,
        identity: ClientIdentity,
        token: str,
        *,
        pow_header: str,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> UploadInfo:
        raw = await self.http.post_multipart(
            self.url(P_FILE_UPLOAD),
            headers=identity.headers(token=token, pow_header=pow_header),
            filename=filename,
            content_type=content_type,
            data=data,
        )
        payload = raw if isinstance(raw, dict) else None
        if not payload or not payload.get("id"):
            raise ProviderError("upload response missing id")
        return UploadInfo(
            id=str(payload["id"]),
            status=str(payload.get("status", "")),
            file_name=str(payload.get("file_name", filename)),
            file_size=int(payload.get("file_size", len(data))),
        )

    async def fetch_files(self, identity: ClientIdentity, token: str,
                          file_ids: list[str]) -> list[FileStatus]:
        data = await self.http.get_json(
            self.url(P_FILE_FETCH),
            headers=identity.headers(token=token),
            params=[("file_ids", ",".join(file_ids))],
        )
        files = data.get("files", []) if isinstance(data, dict) else []
        return [
            FileStatus(
                id=str(item.get("id", "")),
                status=str(item.get("status", "")),
                file_name=str(item.get("file_name", "")),
                file_size=int(item.get("file_size", 0)),
            )
            for item in files
            if isinstance(item, dict)
        ]
