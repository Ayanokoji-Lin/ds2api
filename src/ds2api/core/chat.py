"""Chat orchestration: session planning, PoW, uploads, SSE parsing.

Port of ``ds_core/src/chat.rs`` + ``ds_core/src/chat/request.rs``.

Two timing details are reproduced deliberately:

1. ``_once`` blocks until ``ready`` and at least one data frame have been read.
   That way a muted account or a rate-limit ``hint`` surfaces as a 429 instead of
   a 200 with an empty stream.
2. At EOF the framer is flushed, otherwise a trailing frame without a blank line
   loses ``accumulated_token_usage`` and the final status (Rust issue #93).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..settings import Config
from .endpoints import COMPLETION_TARGET, UPLOAD_TARGET, Endpoints
from .errors import (
    BusinessError,
    ClientError,
    Overloaded,
    ProviderError,
    ProofOfWorkFailed,
    StreamError,
    TERMINAL_BIZ_CODES,
)
from .events import ContentDelta, Done, Meta, StreamEvent, ThinkDelta
from .patch import PatchParser
from .pool import Account, AccountPool
from .sse_framer import SseFramer, parse_frame

logger = logging.getLogger("ds2api.chat")

MAX_ATTEMPTS = 3
RETRY_DELAY = 0.5
UPLOAD_POLL_INTERVAL = 2.0
UPLOAD_POLL_RETRIES = 30
SESSION_IDLE_DEFAULT = 900


@dataclass(slots=True)
class FileUpload:
    filename: str
    content_type: str
    data: bytes


@dataclass(slots=True)
class ChatRequest:
    prompt: str
    model_type: str = "default"
    thinking_enabled: bool = True
    search_enabled: bool = False
    files: list[FileUpload] = field(default_factory=list)
    #: Message fingerprint chain used for reuse matching; None disables reuse.
    delta: tuple[str, ...] | None = None
    #: Text to send instead of ``prompt`` when a cached session is reused.
    incremental_prompt: str | None = None


class Chat:
    def __init__(
        self,
        config: Config,
        pool: AccountPool,
        endpoints: Endpoints,
        pow_solver,
        hif,
    ) -> None:
        self._config = config
        self._pool = pool
        self._endpoints = endpoints
        self._pow = pow_solver
        self._hif = hif
        self._policy = config.ds_core.session_policy
        self._idle_secs = config.ds_core.session_idle_secs or SESSION_IDLE_DEFAULT

    # -- public ------------------------------------------------------------

    async def complete(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            account = await self._pool.acquire()
            ok = False
            try:
                async for event in self._once(account, req):
                    yield event
                ok = True
            except Overloaded:
                raise
            except (ClientError, ProviderError, StreamError, ProofOfWorkFailed) as exc:
                last = exc
                if isinstance(exc, BusinessError) and exc.biz and exc.code in TERMINAL_BIZ_CODES:
                    await self._pool.invalidate(account, str(exc))
            except GeneratorExit:
                ok = True
                raise
            finally:
                await self._pool.release(account, error=not ok)
            if ok:
                return
            if attempt < MAX_ATTEMPTS - 1:
                logger.warning("chat attempt %d failed: %s", attempt + 1, last)
                await asyncio.sleep(RETRY_DELAY)
        raise last or ProviderError("chat failed after retries")

    async def shutdown(self) -> None:
        for account in self._pool.accounts:
            if not account.session_id:
                continue
            try:
                if account.session_parent:
                    await self._endpoints.stop_stream(
                        account.identity, account.token, account.session_id, account.session_parent
                    )
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                pass
            try:
                await self._endpoints.delete_session(
                    account.identity, account.token, account.session_id
                )
            except (Exception, asyncio.CancelledError):  # noqa: BLE001
                pass
            account.session_id = None

    # -- one attempt -------------------------------------------------------

    async def _once(self, account: Account, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        session_id, parent_id, reused = await self._plan_session(account, req)

        # A reused session already holds the history, so only the newest user
        # message is sent, with no ChatML tags.
        prompt_text = (
            req.incremental_prompt
            if reused and req.incremental_prompt is not None
            else req.prompt
        )

        ref_file_ids: list[str] = []
        for upload in req.files:
            ref_file_ids.append(await self._upload(account, upload))

        pow_header = await self._pow_for(account, COMPLETION_TARGET)
        hif_headers = await self._hif.headers_for(account.identity)

        body: dict = {
            "chat_session_id": session_id,
            "model_type": req.model_type,
            "prompt": prompt_text,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": req.thinking_enabled,
            "search_enabled": req.search_enabled,
            "action": None,
            "source": "input",
            "preempt": False,
        }
        if parent_id is not None:
            body["parent_message_id"] = parent_id

        response = await self._endpoints.completion(
            account.identity,
            account.token,
            body,
            pow_header=pow_header,
            hif_headers=hif_headers,
        )

        framer = SseFramer()
        patch = PatchParser()
        stop_id: int | None = None
        finished = False
        yielded_meta = False

        async def frames() -> AsyncIterator[bytes]:
            iterator = response.aiter_content().__aiter__()
            while True:
                try:
                    raw = await anext(iterator)
                except StopAsyncIteration:
                    for tail in framer.flush():
                        yield tail
                    return
                for frame in framer.feed(raw):
                    yield frame

        frame_iter = frames()
        try:
            if not yielded_meta:
                yielded_meta = True
                yield Meta(account.display_id)

            # -- pre-read: ready + at least one data frame -----------------
            pre: list[StreamEvent] = []
            while stop_id is None or not pre:
                try:
                    frame = await anext(frame_iter)
                except StopAsyncIteration:
                    break
                event, data = parse_frame(frame)
                if event == "ready":
                    stop_id = _parse_ready(data)
                    continue
                if event == "hint":
                    _raise_hint(data)
                    continue
                if event == "close":
                    break
                if event in ("update_session", "title"):
                    continue
                if data:
                    pre.extend(_apply_json(patch, data))
            for item in pre:
                yield item

            # -- remaining stream ------------------------------------------
            async for frame in frame_iter:
                event, data = parse_frame(frame)
                if event == "ready":
                    stop_id = _parse_ready(data)
                    continue
                if event == "hint":
                    _raise_hint(data)
                    continue
                if event == "close":
                    break
                if event in ("update_session", "title"):
                    continue
                if not data:
                    continue
                for item in _apply_json(patch, data):
                    yield item

            finish_reason = "stop" if patch.status == "FINISHED" else None
            yield Done(finish_reason, patch.accumulated_token_usage)
            finished = True
        finally:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001
                pass

            if stop_id is not None:
                account.session_parent = stop_id

            # An aborted stream still needs a stop, in both policies.
            if not finished and stop_id is not None:
                await self._safe_stop(account, session_id, stop_id)

            # In reuse mode the session stays alive for the next turn - including
            # the one created by this very request, which is what makes turn 2
            # able to reuse it.
            keep = self._policy == "reuse" and req.delta is not None
            if keep:
                account.session_last_used = time.monotonic()
                account.session_chain = req.delta or ()
            else:
                await self._safe_delete(account, session_id)
                account.session_id = None
                account.session_parent = None

    # -- session planning --------------------------------------------------

    async def _plan_session(
        self, account: Account, req: ChatRequest
    ) -> tuple[str, int | None, bool]:
        if (
            self._policy == "reuse"
            and account.session_id
            and req.delta
            and _chain_matches(account.session_chain, req.delta)
        ):
            logger.debug("reusing session for %s", account.display_id)
            return account.session_id, account.session_parent, True

        session_id = await self._endpoints.create_session(account.identity, account.token)
        account.session_id = session_id
        account.session_parent = None
        account.session_last_used = time.monotonic()
        if req.delta:
            account.session_chain = req.delta
        return session_id, None, False

    async def _safe_stop(self, account: Account, session_id: str, message_id: int) -> None:
        try:
            await self._endpoints.stop_stream(
                account.identity, account.token, session_id, message_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("stop_stream failed: %s", exc)

    async def _safe_delete(self, account: Account, session_id: str) -> None:
        try:
            await self._endpoints.delete_session(
                account.identity, account.token, session_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("delete_session failed: %s", exc)

    # -- pow / upload ------------------------------------------------------

    async def _pow_for(self, account: Account, target_path: str) -> str:
        challenge = await self._endpoints.create_pow_challenge(
            account.identity, account.token, target_path
        )
        result = await self._pow.solve_async(challenge)
        return result.header()

    async def _upload(self, account: Account, upload: FileUpload) -> str:
        pow_header = await self._pow_for(account, UPLOAD_TARGET)
        info = await self._endpoints.upload_file(
            account.identity,
            account.token,
            pow_header=pow_header,
            filename=upload.filename,
            content_type=upload.content_type,
            data=upload.data,
        )
        for _ in range(UPLOAD_POLL_RETRIES):
            files = await self._endpoints.fetch_files(
                account.identity, account.token, [info.id]
            )
            if files:
                status = files[0].status
                if status == "SUCCESS":
                    return info.id
                if status == "FAILED":
                    raise ProviderError(f"file processing failed: {upload.filename}")
            await asyncio.sleep(UPLOAD_POLL_INTERVAL)
        raise ProviderError(f"file processing timed out: {upload.filename}")


# -- helpers ---------------------------------------------------------------


def _chain_matches(cached: tuple[str, ...], incoming: tuple[str, ...]) -> bool:
    """A cached chain is reusable when it is a strict prefix and 1-2 messages behind."""
    if not cached or len(incoming) <= len(cached):
        return False
    if len(incoming) - len(cached) > 2:
        return False
    return incoming[: len(cached)] == cached


def _parse_ready(data: str | None) -> int | None:
    if not data:
        return None
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    value = payload.get("response_message_id")
    return int(value) if isinstance(value, int) else None


def _raise_hint(data: str | None) -> None:
    if not data:
        return
    try:
        payload = json.loads(data)
    except ValueError:
        return
    if payload.get("type") != "error":
        return
    content = str(payload.get("content", "upstream error"))
    finish = str(payload.get("finish_reason", ""))
    if finish == "rate_limit_reached":
        raise Overloaded(content)
    if finish == "input_exceeds_limit":
        raise ProviderError("输入内容超长，请缩短后重试")
    raise ProviderError(content)


def _apply_json(patch: PatchParser, data: str) -> list[StreamEvent]:
    try:
        payload = json.loads(data)
    except ValueError:
        return []
    return patch.apply(payload)
