"""Proof-of-work solver.

Port of ``ds_core/src/accounts/pow.rs``. The hash algorithm lives inside DeepSeek's
wasm blob, so we load the same blob with wasmtime and replay the exact ABI:

    retptr  = __wbindgen_add_to_stack_pointer(-16)
    p, l    = malloc(len, 1) then memory.write(store, bytes, p)
    wasm_solve(retptr, p_ch, l_ch, p_prefix, l_prefix, float(difficulty))
    status  = i32 at retptr        (0 => no solution)
    answer  = f64 at retptr + 8

Verified in Phase 0: difficulty=144000 on a real challenge yields answer=107544,
matching the Rust implementation's recorded output.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import ProofOfWorkFailed

DEFAULT_WASM_URL = (
    "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
)
EXPECTED_ALGORITHM = "DeepSeekHashV1"


@dataclass(slots=True, frozen=True)
class Challenge:
    algorithm: str
    challenge: str
    salt: str
    signature: str
    difficulty: int
    expire_at: int
    target_path: str

    @classmethod
    def from_dict(cls, data: dict) -> Challenge:
        return cls(
            algorithm=str(data.get("algorithm", "")),
            challenge=str(data.get("challenge", "")),
            salt=str(data.get("salt", "")),
            signature=str(data.get("signature", "")),
            difficulty=int(data.get("difficulty", 0)),
            expire_at=int(data.get("expire_at", 0)),
            target_path=str(data.get("target_path", "")),
        )


@dataclass(slots=True, frozen=True)
class PowResult:
    algorithm: str
    challenge: str
    salt: str
    answer: int
    signature: str
    target_path: str

    def header(self) -> str:
        """Base64 header value.

        serde_json's Map is a BTreeMap, so keys are alphabetically ordered and the
        output is compact. Reproducing that byte-for-byte matters only for parity
        checks, but it costs nothing to match.
        """
        payload = {
            "algorithm": self.algorithm,
            "answer": self.answer,
            "challenge": self.challenge,
            "salt": self.salt,
            "signature": self.signature,
            "target_path": self.target_path,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return base64.b64encode(raw).decode("ascii")


def _vname(value: object) -> str:
    text = str(value)
    for name in ("i32", "i64", "f32", "f64", "v128", "funcref", "externref"):
        if name in text:
            return name
    return text


def download_wasm(url: str, cache_path: Path, timeout: float = 60.0) -> bytes:
    """Fetch the wasm blob, preferring an existing cache."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data


class PowSolver:
    """Loads the wasm once and solves challenges on demand."""

    def __init__(self, wasm_bytes: bytes) -> None:
        import wasmtime

        self._wasmtime = wasmtime
        self._engine = wasmtime.Engine()
        self._module = wasmtime.Module(self._engine, wasm_bytes)
        self._names = self._resolve_exports()

    @classmethod
    def from_file(cls, path: Path | str) -> PowSolver:
        return cls(Path(path).read_bytes())

    @classmethod
    def from_url(cls, url: str, cache_path: Path) -> PowSolver:
        return cls(download_wasm(url, cache_path))

    def _resolve_exports(self) -> dict[str, str]:
        module = self._module
        wanted = {
            "add_to_stack": (["i32"], ["i32"], "__wbindgen_add_to_stack_pointer"),
            "alloc": (["i32", "i32"], ["i32"], "__wbindgen_malloc"),
            "solve": (["i32", "i32", "i32", "i32", "i32", "f64"], [], "wasm_solve"),
        }
        resolved: dict[str, str] = {}
        for key, (params, results, preferred) in wanted.items():
            matches: list[str] = []
            for export in module.exports:
                etype = export.type
                eparams = getattr(etype, "params", None)
                if eparams is None:
                    continue
                if [_vname(p) for p in eparams] != params:
                    continue
                if [_vname(r) for r in getattr(etype, "results", [])] != results:
                    continue
                matches.append(export.name)
            if preferred in matches:
                resolved[key] = preferred
            elif key == "alloc":
                fallback = [m for m in matches if m.startswith("__wbindgen_export")]
                if len(fallback) != 1:
                    raise ProofOfWorkFailed(f"cannot resolve alloc export: {matches}")
                resolved[key] = fallback[0]
            elif len(matches) == 1:
                resolved[key] = matches[0]
            else:
                raise ProofOfWorkFailed(
                    f"cannot resolve {key} export params={params}: {matches}"
                )
        return resolved

    def solve(self, challenge: Challenge) -> PowResult:
        if challenge.algorithm != EXPECTED_ALGORITHM:
            raise ProofOfWorkFailed(f"unsupported algorithm: {challenge.algorithm}")

        wasmtime = self._wasmtime
        store = wasmtime.Store(self._engine)
        instance = wasmtime.Instance(store, self._module, [])
        exports = instance.exports(store)

        memory = exports["memory"]
        add_to_stack = exports[self._names["add_to_stack"]]
        alloc = exports[self._names["alloc"]]
        solve_fn = exports[self._names["solve"]]

        prefix = f"{challenge.salt}_{challenge.expire_at}_"
        retptr = add_to_stack(store, -16)

        def write_str(text: str) -> tuple[int, int]:
            raw = text.encode()
            ptr = alloc(store, len(raw), 1)
            # wasmtime >= 40: write(store, value, start)
            memory.write(store, raw, ptr)
            return ptr, len(raw)

        p_ch, l_ch = write_str(challenge.challenge)
        p_pf, l_pf = write_str(prefix)
        difficulty = float(min(max(challenge.difficulty, 0), 0xFFFFFFFF))
        solve_fn(store, retptr, p_ch, l_ch, p_pf, l_pf, difficulty)

        status = int.from_bytes(
            bytes(memory.read(store, retptr, retptr + 4)), "little", signed=True
        )
        value = struct.unpack("<d", bytes(memory.read(store, retptr + 8, retptr + 16)))[0]
        if status == 0:
            raise ProofOfWorkFailed("wasm reported no solution")

        return PowResult(
            algorithm=challenge.algorithm,
            challenge=challenge.challenge,
            salt=challenge.salt,
            answer=int(value),
            signature=challenge.signature,
            target_path=challenge.target_path,
        )

    async def solve_async(self, challenge: Challenge) -> PowResult:
        # wasm execution is synchronous and CPU-bound; never block the loop.
        return await asyncio.to_thread(self.solve, challenge)
