"""Inline file extraction.

The upstream accepts ``ref_file_ids`` pointing at previously uploaded files. We
only handle inline ``data:`` URLs; remote http(s) image URLs cannot be fetched
server-side, so they are converted into a "please visit this link" prompt fragment
and force search mode on (same behaviour as the Rust project).
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from ...core.errors import BadRequest
from ..types import ChatCompletionsRequest

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "application/pdf": "pdf",
    "text/plain": "txt",
}


@dataclass(slots=True)
class FilePayload:
    filename: str
    content_type: str
    data: bytes


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    if not payload:
        return None
    meta = header[5:]
    content_type = meta.split(";", 1)[0] or "application/octet-stream"
    if ";base64" in meta:
        try:
            return content_type, base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest(f"invalid base64 data URL: {exc}") from exc
    return content_type, payload.encode()


def extract(req: ChatCompletionsRequest) -> tuple[list[FilePayload], bool]:
    """Return ``(files, has_http_urls)`` for the whole message list."""
    files: list[FilePayload] = []
    has_http_urls = False

    for message in req.messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if part.type == "file" and part.file and part.file.file_data:
                decoded = _decode_data_url(part.file.file_data)
                if decoded is None:
                    raise BadRequest("file.file_data must be a data: URL")
                content_type, data = decoded
                name = part.file.filename or f"file.{_EXT_BY_MIME.get(content_type, 'bin')}"
                files.append(FilePayload(name, content_type, data))
            elif part.type == "image_url" and part.image_url:
                url = part.image_url.url
                if url.startswith(("http://", "https://")):
                    has_http_urls = True
                    continue
                decoded = _decode_data_url(url)
                if decoded is None:
                    continue
                content_type, data = decoded
                ext = _EXT_BY_MIME.get(content_type, "bin")
                files.append(FilePayload(f"image.{ext}", content_type, data))

    return files, has_http_urls
