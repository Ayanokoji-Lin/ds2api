"""Phase 0 spike: verify wasmtime can run DeepSeek's PoW wasm.

This answers the single most important feasibility question: can we reproduce
``ds_core/src/accounts/pow.rs`` in pure Python by loading the same wasm?

Run:  python tools/spike_wasm.py
"""

from __future__ import annotations

import base64
import json
import struct
import sys
import urllib.request
from pathlib import Path

WASM_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "sha3_wasm_bg.wasm"

# Captured from a real /chat/create_pow_challenge response (docs/development.md).
REAL_CHALLENGE = {
    "algorithm": "DeepSeekHashV1",
    "challenge": "7ffc9d19b6eed96a6fca68f8ffe30ee61035d4959e4180f187bf85b356016a96",
    "salt": "3bde54628ea8413fee87",
    "signature": "ce4678cf7a1290c2a7ac88c4195a5b8497e5fc4b0e8044e804f5a6f3af6fe462",
    "difficulty": 144000,
    "expire_at": 1775380966945,
    "target_path": "/api/v0/chat/completion",
}


def fetch_wasm() -> bytes:
    if CACHE.exists() and CACHE.stat().st_size > 0:
        print(f"[wasm] using cache {CACHE} ({CACHE.stat().st_size} bytes)")
        return CACHE.read_bytes()
    print(f"[wasm] downloading {WASM_URL}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(WASM_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    CACHE.write_bytes(data)
    print(f"[wasm] saved {len(data)} bytes -> {CACHE}")
    return data


def vname(v: object) -> str:
    """Normalise a wasmtime ValType to a short string like 'i32'."""
    s = str(v)
    for k in ("i32", "i64", "f32", "f64", "v128", "funcref", "externref"):
        if k in s:
            return k
    return s


def dump_exports(module: object) -> None:
    print("[wasm] exports:")
    for e in module.exports:  # type: ignore[attr-defined]
        t = e.type
        params = getattr(t, "params", None)
        if params is None:
            print(f"  - {e.name}: {t}")
            continue
        results = getattr(t, "results", [])
        print(
            f"  - {e.name}: ({', '.join(vname(p) for p in params)})"
            f" -> ({', '.join(vname(r) for r in results)})"
        )


def pick_func(module: object, want_params: list[str], want_results: list[str],
              preferred: str, fallback_prefix: str = "") -> str:
    """Locate an export by signature, preferring the canonical wbindgen name."""
    matches: list[str] = []
    for e in module.exports:  # type: ignore[attr-defined]
        t = e.type
        params = getattr(t, "params", None)
        if params is None:
            continue
        if [vname(p) for p in params] != want_params:
            continue
        if [vname(r) for r in getattr(t, "results", [])] != want_results:
            continue
        matches.append(e.name)
    if preferred in matches:
        return preferred
    if fallback_prefix:
        for name in matches:
            if name.startswith(fallback_prefix):
                return name
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(
        f"cannot resolve export params={want_params} results={want_results}; "
        f"candidates={matches}"
    )


def solve(store, exports, names: dict[str, str], challenge: dict) -> dict:
    add_to_stack = exports[names["add_to_stack"]]
    alloc = exports[names["alloc"]]
    wasm_solve = exports[names["solve"]]
    memory = exports["memory"]

    prefix = f"{challenge['salt']}_{challenge['expire_at']}_"
    retptr = add_to_stack(store, -16)

    def write_str(text: str) -> tuple[int, int]:
        raw = text.encode("utf-8")
        ptr = alloc(store, len(raw), 1)
        # wasmtime >= 40: write(store, value, start)
        memory.write(store, raw, ptr)
        return ptr, len(raw)

    p_ch, l_ch = write_str(challenge["challenge"])
    p_pf, l_pf = write_str(prefix)

    difficulty = float(min(int(challenge["difficulty"]), 0xFFFFFFFF))
    wasm_solve(store, retptr, p_ch, l_ch, p_pf, l_pf, difficulty)

    status = int.from_bytes(memory.read(store, retptr, retptr + 4), "little", signed=True)
    value = struct.unpack("<d", memory.read(store, retptr + 8, retptr + 16))[0]
    if status == 0:
        raise RuntimeError("wasm reported no solution")

    payload = {
        "algorithm": challenge["algorithm"],
        "challenge": challenge["challenge"],
        "salt": challenge["salt"],
        "answer": int(value),
        "signature": challenge["signature"],
        "target_path": challenge["target_path"],
    }
    header = base64.b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return {"status": status, "value": value, "answer": int(value), "header": header}


def main() -> int:
    try:
        import wasmtime
    except ImportError:
        print("FAIL: wasmtime not installed")
        return 1

    print(f"[wasmtime] version {getattr(wasmtime, '__version__', '?')}")
    wasm_bytes = fetch_wasm()

    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, wasm_bytes)
    dump_exports(module)

    names = {
        "add_to_stack": pick_func(module, ["i32"], ["i32"], "__wbindgen_add_to_stack_pointer"),
        "alloc": pick_func(module, ["i32", "i32"], ["i32"], "__wbindgen_malloc", "__wbindgen_export"),
        "solve": pick_func(
            module,
            ["i32", "i32", "i32", "i32", "i32", "f64"],
            [],
            "wasm_solve",
        ),
    }
    print(f"[wasm] resolved exports: {names}")

    results = []
    for attempt in (1, 2):
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        exports = instance.exports(store)
        out = solve(store, exports, names, REAL_CHALLENGE)
        print(
            f"[solve #{attempt}] status={out['status']} value={out['value']!r} "
            f"answer={out['answer']}"
        )
        print(f"[solve #{attempt}] header={out['header']}")
        results.append(out["answer"])

    ok = results[0] == results[1] and isinstance(results[0], int)
    print(f"[determinism] {results} -> {'OK' if ok else 'MISMATCH'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
