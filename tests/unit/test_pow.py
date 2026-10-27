"""PoW solver tests.

These run against the cached wasm downloaded during Phase 0 and assert the exact
answer recorded by the Rust implementation (answer=107544 for this challenge).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ds2api.core.pow import Challenge, PowSolver, PowResult

WASM = Path(__file__).resolve().parents[2] / "data" / "cache" / "sha3_wasm_bg.wasm"

REAL = Challenge(
    algorithm="DeepSeekHashV1",
    challenge="7ffc9d19b6eed96a6fca68f8ffe30ee61035d4959e4180f187bf85b356016a96",
    salt="3bde54628ea8413fee87",
    signature="ce4678cf7a1290c2a7ac88c4195a5b8497e5fc4b0e8044e804f5a6f3af6fe462",
    difficulty=144000,
    expire_at=1775380966945,
    target_path="/api/v0/chat/completion",
)

pytestmark = pytest.mark.skipif(not WASM.exists(), reason="wasm not cached")


@pytest.fixture(scope="module")
def solver() -> PowSolver:
    return PowSolver.from_file(WASM)


def test_solves_real_challenge(solver: PowSolver) -> None:
    result = solver.solve(REAL)
    assert isinstance(result, PowResult)
    assert result.answer == 107544
    assert result.algorithm == "DeepSeekHashV1"


def test_deterministic(solver: PowSolver) -> None:
    assert solver.solve(REAL).answer == solver.solve(REAL).answer


def test_header_is_base64_of_sorted_compact_json(solver: PowSolver) -> None:
    import base64
    import json

    header = solver.solve(REAL).header()
    raw = base64.b64decode(header).decode()
    payload = json.loads(raw)
    assert list(payload) == sorted(payload), "keys must be alphabetically ordered"
    assert " " not in raw, "must be compact separators"
    assert payload["answer"] == 107544
    assert payload["target_path"] == "/api/v0/chat/completion"


def test_rejects_unknown_algorithm(solver: PowSolver) -> None:
    from dataclasses import replace

    from ds2api.core.errors import ProofOfWorkFailed

    with pytest.raises(ProofOfWorkFailed):
        solver.solve(replace(REAL, algorithm="Nope"))


def test_challenge_from_dict() -> None:
    challenge = Challenge.from_dict(
        {
            "algorithm": "DeepSeekHashV1",
            "challenge": "abc",
            "salt": "s",
            "signature": "sig",
            "difficulty": "1000",
            "expire_at": 123,
            "target_path": "/api/v0/chat/completion",
        }
    )
    assert challenge.difficulty == 1000
    assert challenge.expire_at == 123
