"""Phase 0 spike: verify curl_cffi can reproduce the 17-header browser identity.

Two checks:
  1. Hit a local echo server and diff the headers actually sent against the
     header set extracted from ``ds_core/src/accounts/client.rs``.
  2. Query tls.peet.ws to record the JA3/JA4 fingerprint for the archive.

Run:  python tools/spike_curl.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

# The exact set from ClientIdentity::headers(), in the order the Rust code sets them.
IDENTITY_HEADERS = {
    "User-Agent": DEFAULT_UA,
    "X-Client-Version": "2.5.0",
    "X-Client-Platform": "web",
    "X-Client-Locale": "zh_CN",
    "X-Client-Bundle-Id": "com.deepseek.chat",
    "X-Device-Id": "11111111-2222-4333-8444-555555555555",
    "X-Device-Model": "",
    "X-Client-Timezone-Offset": "28800",
    "Accept-Language": "zh-CN",
    "Referer": "https://chat.deepseek.com/",
    "Accept": "*/*",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": '"Chromium";v="140", "Not_A Brand";v="24", "Google Chrome";v="140"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    # curl_cffi's chrome136 profile defaults to a top-level navigation shape and
    # adds these two. The real DeepSeek frontend issues XHR/fetch requests, which
    # never carry them. Passing None removes the header entirely (verified).
    "Sec-Fetch-User": None,
    "Upgrade-Insecure-Requests": None,
}

# Headers that must NOT be present (the Rust client never sends them).
FORBIDDEN = ["origin", "cookie", "priority"]

CANDIDATE_PROFILES = ["chrome136", "chrome133a", "chrome131", "chrome124", "chrome120", "chrome"]


class EchoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps(
            {"headers": {k.lower(): v for k, v in self.headers.items()}}, ensure_ascii=False
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def start_echo() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address[:2]
    return srv, f"http://{host}:{port}/echo"


def pick_profile() -> str | None:
    from curl_cffi.requests import AsyncSession

    for name in CANDIDATE_PROFILES:
        try:
            AsyncSession(impersonate=name)
            print(f"[profile] '{name}' accepted")
            return name
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] '{name}' rejected: {type(exc).__name__}: {exc}")
    return None


async def check_headers(profile: str, url: str) -> bool:
    from curl_cffi.requests import AsyncSession

    async with AsyncSession(impersonate=profile) as session:
        resp = await session.get(url, headers=IDENTITY_HEADERS, timeout=15)
        got = resp.json()["headers"]

    print("\n[headers] received by echo server:")
    for key in sorted(got):
        print(f"  {key}: {got[key]!r}")

    print("\n[headers] diff vs expectation:")
    ok = True
    for key, expected in IDENTITY_HEADERS.items():
        lk = key.lower()
        if expected is None:
            # None means "must be absent"
            if lk in got:
                print(f"  LEAKED   {lk}: {got[lk]!r}  (must be removed)")
                ok = False
            else:
                print(f"  REMOVED  {lk}")
            continue
        if lk not in got:
            print(f"  MISSING  {lk} (expected {expected!r})")
            ok = False
            continue
        if got[lk] != expected:
            print(f"  DIFF     {lk}: got {got[lk]!r} expected {expected!r}")
            # header casing/ordering differences are tolerable, value differences are not
            ok = False
        else:
            print(f"  OK       {lk}")

    for bad in FORBIDDEN:
        if bad in got:
            print(f"  LEAKED   {bad}: {got[bad]!r}  (must not be sent)")
            ok = False

    extra = sorted(set(got) - {k.lower() for k in IDENTITY_HEADERS})
    print(f"\n[headers] extra headers curl_cffi added: {extra}")
    return ok


async def check_tls(profile: str) -> None:
    from curl_cffi.requests import AsyncSession

    try:
        async with AsyncSession(impersonate=profile) as session:
            resp = await session.get("https://tls.peet.ws/api/all", timeout=25)
            data = resp.json()
        out = {
            "ja3": data.get("tls", {}).get("ja3_hash"),
            "ja3_text": data.get("tls", {}).get("ja3"),
            "ja4": data.get("tls", {}).get("ja4"),
            "http2": (data.get("http2") or {}).get("akamai_fingerprint"),
            "user_agent": data.get("user_agent"),
        }
        print("\n[tls] fingerprint:")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        path = __file__.replace("spike_curl.py", "..").rstrip("\\/")
        from pathlib import Path

        archive = Path(__file__).resolve().parent.parent / "docs" / "tls-fingerprint.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps({"profile": profile, **out}, indent=2), encoding="utf-8")
        print(f"[tls] archived -> {archive}")
    except Exception as exc:  # noqa: BLE001
        print(f"[tls] SKIP (network unavailable): {type(exc).__name__}: {exc}")


async def main() -> int:
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        print("FAIL: curl_cffi not installed")
        return 1

    print(f"[curl_cffi] version {getattr(curl_cffi, '__version__', '?')}")
    profile = pick_profile()
    if not profile:
        print("FAIL: no usable impersonate profile")
        return 1

    srv, url = start_echo()
    try:
        ok = await check_headers(profile, url)
        await check_tls(profile)
    finally:
        srv.shutdown()

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
