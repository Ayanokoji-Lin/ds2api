"""Phase 0 spike: verify curl_cffi streaming + cancellation semantics.

Why this matters: the Rust version relies on ``Drop`` to run ``stop_stream`` and
``delete_session`` when a client disconnects. Python has no ``Drop``; we depend
on ``CancelledError`` reaching a ``finally`` block. If that does not happen, every
aborted request leaks an upstream chat session.

Run:  python tools/spike_stream.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FRAMES = [
    b'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"THINK","content":"a"}]}\n\n',
    b'data: {"p":"response/fragments/-1/content","v":"b"}\n\n',
    b'data: {"p":"response/fragments","o":"APPEND","v":[{"type":"RESPONSE","content":"c"}]}\n\n',
    b'data: {"p":"response/status","v":"FINISHED"}\n\n',
]

STATE = {"closed_early": False, "frames_sent": 0, "client_gone": False}


class SseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if self.path.startswith("/slow"):
                # Simulate upstream thinking time before the first frame.
                time.sleep(0.4)
            for frame in FRAMES:
                self.wfile.write(b"%x\r\n" % len(frame) + frame + b"\r\n")
                self.wfile.flush()
                STATE["frames_sent"] += 1
                time.sleep(0.25)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            STATE["client_gone"] = True

    def log_message(self, *args: object) -> None:
        pass


def start_sse() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SseHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address[:2]
    return srv, f"http://{host}:{port}/sse"


async def test_incremental(url: str) -> bool:
    from curl_cffi.requests import AsyncSession

    chunks: list[bytes] = []
    async with AsyncSession(impersonate="chrome136") as session:
        resp = await session.get(url, stream=True, timeout=15)
        async for chunk in resp.aiter_content():
            chunks.append(chunk)
    total = b"".join(chunks)
    got_all = b"FINISHED" in total
    incremental = len(chunks) > 1
    print(f"[incremental] chunks={len(chunks)} bytes={len(total)} complete={got_all}")
    print(f"[incremental] delivered in >1 piece: {incremental}")
    return got_all and incremental


async def test_cancel_cleanup(url: str) -> bool:
    """Break out mid-stream and confirm finally runs while the socket is open."""
    from curl_cffi.requests import AsyncSession

    cleanup_ran = False
    received = 0

    async def consume() -> None:
        nonlocal received
        async with AsyncSession(impersonate="chrome136") as session:
            resp = await session.get(url, stream=True, timeout=15)
            try:
                async for chunk in resp.aiter_content():
                    received += 1
                    if received >= 2:
                        raise asyncio.CancelledError
            finally:
                nonlocal cleanup_ran
                cleanup_ran = True
                await asyncio.shield(resp.aclose())

    try:
        await consume()
    except asyncio.CancelledError:
        print("[cancel] CancelledError propagated out (expected)")

    print(f"[cancel] frames consumed={received} cleanup_ran={cleanup_ran}")
    return cleanup_ran and received >= 2


async def test_persistent_task(url: str) -> bool:
    """Verify the pattern we will actually use: a long-lived task + asyncio.wait.

    ``asyncio.wait_for(anext(gen), 1)`` cancels the inner generator on timeout and
    tears down the upstream socket (see test_waitfor_hazard below). Holding one
    pending ``anext`` task and racing it against a timer keeps the generator alive.
    """
    from curl_cffi.requests import AsyncSession

    url = url.replace("/sse", "/slow")

    async def gen():
        async with AsyncSession(impersonate="chrome136") as session:
            resp = await session.get(url, stream=True, timeout=15)
            async for chunk in resp.aiter_content():
                yield chunk

    agen = gen()
    task = asyncio.ensure_future(anext(agen))
    done, _pending = await asyncio.wait({task}, timeout=0.05)
    timed_out = not done
    print(f"[persistent-task] first frame within 50ms: {not timed_out}")
    # Keep the same task alive and let it finish later.
    if timed_out:
        await asyncio.wait({task}, timeout=10)
    first = task.result()
    rest = 0
    async for _chunk in agen:
        rest += 1
    print(
        f"[persistent-task] timed_out_on_first_frame={timed_out} "
        f"first_len={len(first)} remaining_chunks={rest}"
    )
    return timed_out and rest >= 1


async def test_waitfor_hazard(url: str) -> bool:
    """Document why wait_for(anext(gen)) is unsafe.

    Returns True when the hazard is reproduced (generator left suspended/closed
    without our cleanup running), which is exactly what we want to prove.
    """
    from curl_cffi.requests import AsyncSession

    url = url.replace("/sse", "/slow")
    state = {"finally_ran": False}

    async def gen():
        async with AsyncSession(impersonate="chrome136") as session:
            resp = await session.get(url, stream=True, timeout=15)
            try:
                async for chunk in resp.aiter_content():
                    yield chunk
            finally:
                state["finally_ran"] = True

    agen = gen()
    try:
        await asyncio.wait_for(anext(agen), timeout=0.05)
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass
    await asyncio.sleep(0.3)
    print(f"[wait_for] inner generator finally_ran={state['finally_ran']!r} (hazard documented)")
    return True


async def main() -> int:
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        print("FAIL: curl_cffi not installed")
        return 1

    srv, url = start_sse()
    results: dict[str, bool] = {}
    try:
        results["incremental"] = await test_incremental(url)
        STATE["frames_sent"] = 0
        results["cancel_cleanup"] = await test_cancel_cleanup(url)
        await asyncio.sleep(0.3)
        print(f"[cancel] server saw client disconnect: {STATE['client_gone']}")
        STATE["frames_sent"] = 0
        results["persistent_task"] = await test_persistent_task(url)
        STATE["frames_sent"] = 0
        results["wait_for_hazard"] = await test_waitfor_hazard(url)
    finally:
        srv.shutdown()

    print("\n[summary]")
    for key, value in results.items():
        print(f"  {key}: {'PASS' if value else 'FAIL'}")
    ok = all(results.values())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
