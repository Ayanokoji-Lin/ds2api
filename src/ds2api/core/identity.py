"""Client identity: the exact header set and deterministic device UUIDs.

Port of ``ClientIdentity`` and ``derive_device_uuid`` from
``ds_core/src/accounts/client.rs``.

Two details matter and are easy to get wrong:

1. The real DeepSeek frontend issues XHR/fetch calls. curl_cffi's ``chrome136``
   profile defaults to a top-level-navigation shape and adds ``Sec-Fetch-User``
   and ``Upgrade-Insecure-Requests``; both must be removed by passing ``None``.
2. ``derive_device_uuid`` uses ``FNV_OFFSET_BASIS ^ offset`` as the seed (not the
   offset itself) and serialises both halves **big-endian**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

SEC_CH_UA = '"Chromium";v="140", "Not_A Brand";v="24", "Google Chrome";v="140"'

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
GOLDEN_RATIO = 0x9E3779B97F4A7C15

_U64 = 0xFFFFFFFFFFFFFFFF


def _fnv1a(data: bytes, offset: int) -> int:
    h = (FNV_OFFSET_BASIS ^ offset) & _U64
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & _U64
    return h


def derive_device_uuid(seed: bytes) -> str:
    """Derive a stable RFC 4122 v4-shaped UUID from ``seed``."""
    hi = _fnv1a(seed, 0)
    lo = _fnv1a(seed, GOLDEN_RATIO)
    raw = bytearray(hi.to_bytes(8, "big") + lo.to_bytes(8, "big"))
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # variant 10xx
    hexs = raw.hex()
    return f"{hexs[0:8]}-{hexs[8:12]}-{hexs[12:16]}-{hexs[16:20]}-{hexs[20:32]}"


def account_device_uuid(device_id: str) -> str:
    """Per-account X-Device-Id, derived from the Shumei fingerprint."""
    return derive_device_uuid(f"ds-free-api:x-device-id:{device_id.strip()}".encode())


#: Headers only a desktop browser sends. The native app (OkHttp) sends none of
#: them, and mixing an Android UA with Chrome client hints is a combination no
#: real client produces - the upstream's own notes call this out.
_BROWSER_ONLY_HEADERS = (
    "Sec-Ch-Ua",
    "Sec-Ch-Ua-Mobile",
    "Sec-Ch-Ua-Platform",
    "Sec-Fetch-Dest",
    "Sec-Fetch-Mode",
    "Sec-Fetch-Site",
    "Sec-Fetch-User",
    "Upgrade-Insecure-Requests",
    "Priority",
)


@dataclass(slots=True)
class ClientIdentity:
    """The browser persona sent on every upstream request."""

    user_agent: str = DEFAULT_USER_AGENT
    client_version: str = "2.5.0"
    client_platform: str = "android"
    client_locale: str = "zh_CN"
    client_bundle_id: str = "com.deepseek.chat"
    device_id: str = ""
    device_model: str = "Pixel 7"
    timezone_offset: str = "28800"
    referer: str = "https://chat.deepseek.com/"
    #: Value sent as ``os`` in the login payload. Must agree with the UA.
    client_os: str = "android"
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_app(self) -> bool:
        return self.client_platform.lower() in ("android", "ios")

    def headers(
        self,
        *,
        site: str = "same-origin",
        login: bool = False,
        token: str | None = None,
        pow_header: str | None = None,
        hif: dict[str, str] | None = None,
    ) -> dict[str, str | None]:
        """Build the header set.

        Values of ``None`` mean "delete this header", which curl_cffi honours.
        """
        h: dict[str, str | None] = {
            "User-Agent": self.user_agent,
            "X-Client-Version": self.client_version,
            "X-Client-Platform": self.client_platform,
            "X-Client-Locale": self.client_locale,
            "X-Client-Bundle-Id": self.client_bundle_id,
            "X-Device-Id": self.device_id,
            "X-Device-Model": self.device_model,
            "X-Client-Timezone-Offset": self.timezone_offset,
            "Accept-Language": self.client_locale.replace("_", "-"),
            "Referer": "https://chat.deepseek.com/sign_in" if login else self.referer,
            "Accept": "*/*",
        }

        if self.is_app:
            # Native app shape: no client hints, no sec-fetch-*. curl_cffi's
            # chrome profile injects them, so remove them explicitly.
            for name in _BROWSER_ONLY_HEADERS:
                h[name] = None
        else:
            h.update(
                {
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": site,
                    "Sec-Ch-Ua": SEC_CH_UA,
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": '"Windows"',
                    # Navigation-only headers the XHR frontend never sends.
                    "Sec-Fetch-User": None,
                    "Upgrade-Insecure-Requests": None,
                }
            )

        h.update(self.extra)
        if token:
            h["Authorization"] = f"Bearer {token}"
        if pow_header:
            h["X-Ds-Pow-Response"] = pow_header
        if hif:
            h.update(hif)
        return h
