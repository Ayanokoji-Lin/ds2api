"""Account pool: one account = one session = one concurrent request.

Port of ``ds_core/src/accounts/pool.rs``.

Key rules:

* pick the least-recently-used idle account whose hourly quota is not exhausted
* wait up to 30s for a slot, then raise :class:`Overloaded` (-> HTTP 429)
* a failed request marks the account ERROR; three consecutive failures make it
  INVALID and stop retries
* terminal business codes (muted / banned / risk device) go straight to INVALID -
  retrying only extends the penalty
* a background task re-logins ERROR accounts every 60s
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from ..settings import AccountConfig, Config
from .errors import BusinessError, CoreError, Overloaded, TERMINAL_BIZ_CODES
from .identity import ClientIdentity, account_device_uuid, derive_device_uuid

logger = logging.getLogger("ds2api.pool")

IDLE = "idle"
BUSY = "busy"
ERROR = "error"
INVALID = "invalid"

MAX_ERROR_COUNT = 3
QUOTA_WINDOW_SECS = 3600
ACQUIRE_TIMEOUT = 30.0
ACQUIRE_POLL = 0.2
REAPER_INTERVAL = 60.0


@dataclass(slots=True)
class Account:
    key: str
    config: AccountConfig
    identity: ClientIdentity
    token: str = ""
    state: str = IDLE
    error_count: int = 0
    window: deque[float] = field(default_factory=deque)
    last_used: float = 0.0

    # session reuse state
    session_id: str | None = None
    session_parent: int | None = None
    session_last_used: float = 0.0
    session_chain: tuple[str, ...] = ()

    @property
    def display_id(self) -> str:
        raw = self.config.identifier or self.key
        return raw[:3] + "***" if len(raw) > 3 else "***"


def _build_identity(config: Config, account: AccountConfig) -> ClientIdentity:
    core = config.ds_core
    if core.client_device_id:
        device_id = core.client_device_id
    elif account.device_id.strip():
        device_id = account_device_uuid(account.device_id)
    else:
        # No per-account fingerprint: fall back to a stable per-deployment id.
        device_id = derive_device_uuid(f"ds-free-api:api-base:{core.api_base}".encode())

    platform = core.client_platform
    os_value = core.client_os
    user_agent = core.user_agent
    device_model = core.client_device_model

    # The upstream closed web password login: os="web" now returns a bare
    # 422 {"detail":[{"loc":"body"}]}. Downgrade to the app identity so existing
    # configs keep working, but say so loudly.
    if os_value.strip().lower() == "web":
        logger.warning(
            "client_os=\"web\" is rejected by the upstream (422); using the android "
            "identity instead. Update config.toml: client_os/client_platform = \"android\" "
            "and user_agent = \"DeepSeek/%s Android/35\"",
            core.client_version,
        )
        os_value = "android"
        platform = "android"
        if user_agent.startswith("Mozilla"):
            user_agent = f"DeepSeek/{core.client_version} Android/35"
        if not device_model:
            device_model = "Pixel 7"

    return ClientIdentity(
        user_agent=user_agent,
        client_version=core.client_version,
        client_platform=platform,
        client_locale=core.client_locale,
        client_bundle_id=core.client_bundle_id,
        device_id=device_id,
        device_model=device_model,
        timezone_offset=core.client_timezone_offset,
        client_os=os_value,
    )


class AccountPool:
    def __init__(self, config: Config, endpoints) -> None:
        self._config = config
        self._endpoints = endpoints
        self._accounts: list[Account] = []
        self._cond = asyncio.Condition()
        self._reaper: asyncio.Task | None = None
        self._quota = config.ds_core.hourly_request_quota

        seen: set[str] = set()
        for index, account in enumerate(config.ds_core.accounts):
            # Guard against an easy misconfiguration: pasting the email into the
            # mobile field makes the upstream reject the whole login body with a
            # bare 422, which reads like a serialisation bug.
            if account.mobile and "@" in account.mobile:
                logger.warning(
                    "account %s: mobile looks like an email address; ignoring it "
                    "and logging in with the email field only",
                    (account.identifier or f"#{index}")[:3] + "***",
                )
                account.mobile = ""

            key = account.identifier or f"account-{index}"
            if key in seen:
                logger.warning("duplicate account %s skipped", key[:3] + "***")
                continue
            seen.add(key)
            self._accounts.append(
                Account(key=key, config=account, identity=_build_identity(config, account))
            )

        device_ids = [a.identity.device_id for a in self._accounts]
        if len(device_ids) != len(set(device_ids)):
            logger.warning(
                "multiple accounts share one device identity; upstream may link them "
                "and mute the group. Give each account its own device_id."
            )

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        for account in self._accounts:
            try:
                await self._login(account)
            except BusinessError as exc:
                if exc.biz and exc.code in TERMINAL_BIZ_CODES:
                    account.state = INVALID
                    logger.error("account %s is unusable: %s", account.display_id, exc)
                else:
                    account.state = ERROR
                    logger.warning("login failed for %s: %s", account.display_id, exc)
            except Exception as exc:  # noqa: BLE001
                account.state = ERROR
                logger.warning("login failed for %s: %s", account.display_id, exc)

        usable = [a for a in self._accounts if a.state == IDLE]
        if not usable:
            logger.error("no usable accounts; /v1 requests will return 429")
        else:
            logger.info("%d/%d accounts ready", len(usable), len(self._accounts))

        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reaper_loop())

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            try:
                await self._reaper
            except asyncio.CancelledError:
                pass
            self._reaper = None

    async def _login(self, account: Account) -> None:
        cfg = account.config
        info = await self._endpoints.login(
            account.identity,
            email=cfg.email,
            mobile=cfg.mobile,
            password=cfg.password,
            area_code=cfg.area_code,
            device_id=cfg.device_id,
            os_name=account.identity.client_os,
        )
        account.token = info.token
        try:
            rotated = await self._endpoints.check_device(account.identity, info.token)
            if rotated:
                account.token = rotated
        except CoreError as exc:
            # Non-fatal: the Rust client logs and continues.
            logger.debug("check_device failed for %s: %s", account.display_id, exc)
        account.state = IDLE
        account.error_count = 0
        logger.info("account %s logged in", account.display_id)

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(REAPER_INTERVAL)
            for account in list(self._accounts):
                if account.state != ERROR:
                    continue
                try:
                    await self._login(account)
                except BusinessError as exc:
                    if exc.biz and exc.code in TERMINAL_BIZ_CODES:
                        account.state = INVALID
                        logger.error("account %s marked invalid: %s", account.display_id, exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("relogin failed for %s: %s", account.display_id, exc)

    # -- scheduling --------------------------------------------------------

    def _quota_ok(self, account: Account, now: float) -> bool:
        if self._quota <= 0:
            return True
        window = account.window
        while window and now - window[0] > QUOTA_WINDOW_SECS:
            window.popleft()
        return len(window) < self._quota

    def _pick(self) -> Account | None:
        now = time.monotonic()
        candidates = [
            a for a in self._accounts if a.state == IDLE and self._quota_ok(a, now)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda a: a.last_used)

    async def acquire(self, timeout: float = ACQUIRE_TIMEOUT) -> Account:
        # An empty or fully-dead pool can never recover within the wait window,
        # so fail fast instead of making the client wait 30s.
        if not self._accounts:
            raise Overloaded("no accounts configured")
        if all(a.state == INVALID for a in self._accounts):
            raise Overloaded("all accounts are invalid (muted, banned or bad credentials)")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        async with self._cond:
            while True:
                account = self._pick()
                if account is not None:
                    account.state = BUSY
                    account.last_used = loop.time()
                    if self._quota > 0:
                        account.window.append(loop.time())
                    return account
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise Overloaded("no account available (all busy, rate-limited or invalid)")
                try:
                    await asyncio.wait_for(self._cond.wait(), min(remaining, ACQUIRE_POLL))
                except asyncio.TimeoutError:
                    pass

    async def release(self, account: Account, *, error: bool = False) -> None:
        async with self._cond:
            if account.state == BUSY:
                if error:
                    account.error_count += 1
                    if account.error_count >= MAX_ERROR_COUNT:
                        account.state = INVALID
                        logger.error("account %s invalidated after %d failures",
                                     account.display_id, account.error_count)
                    else:
                        account.state = ERROR
                else:
                    account.error_count = 0
                    account.state = IDLE
            self._cond.notify_all()

    async def invalidate(self, account: Account, reason: str = "") -> None:
        async with self._cond:
            account.state = INVALID
            logger.error("account %s invalidated: %s", account.display_id, reason)
            self._cond.notify_all()

    # -- introspection -----------------------------------------------------

    @property
    def accounts(self) -> list[Account]:
        return list(self._accounts)

    def counts(self) -> dict[str, int]:
        out = {IDLE: 0, BUSY: 0, ERROR: 0, INVALID: 0}
        for account in self._accounts:
            out[account.state] = out.get(account.state, 0) + 1
        return out

    def statuses(self) -> list[dict]:
        now = time.monotonic()
        result = []
        for account in self._accounts:
            self._quota_ok(account, now)
            result.append(
                {
                    "id": account.display_id,
                    "state": account.state,
                    "error_count": account.error_count,
                    "requests_in_window": len(account.window),
                    "quota": self._quota,
                    "window_seconds": QUOTA_WINDOW_SECS,
                    "has_session": account.session_id is not None,
                }
            )
        return result
