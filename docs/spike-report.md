# Phase 0 spike report

Date: 2026-10-27
Environment: Windows AMD64, Python 3.13.14 (managed venv `envs/ds2api`)

## Verdict

**All three spikes PASS.** Proceed to Phase 1.

Installed versions differ from the plan's pins but work:

| Package | Planned | Installed |
|---|---|---|
| curl-cffi | >=0.11,<0.14 | 0.17.0 |
| wasmtime | >=27,<36 | 50.0.1 |
| tiktoken | >=0.8,<1 | 0.14.0 |
| fastapi | >=0.115,<1.0 | 0.162.0 |
| pydantic | >=2.9,<3 | 2.14.1 |

## 1. PoW wasm (`tools/spike_wasm.py`) — PASS

wasm downloaded: 26,612 bytes.

Exports discovered (signature-probed, not name-hardcoded):

```
memory
wasm_deepseek_hash_v1            (i32, i32, i32) -> ()
wasm_solve                       (i32, i32, i32, i32, i32, f64) -> ()
__wbindgen_add_to_stack_pointer  (i32) -> (i32)
__wbindgen_export_0              (i32, i32) -> (i32)     <- malloc
__wbindgen_export_1              (i32, i32, i32, i32) -> (i32)
__wbindgen_export_2              (i32, i32, i32) -> ()
```

Real challenge (`salt=3bde54628ea8413fee87`, `expire_at=1775380966945`, `difficulty=144000`)
solves to `status=1`, `value=107544.0`, `answer=107544`, deterministic across two runs.

`107544` matches the value recorded in the Rust project's `docs/development.md`,
which is strong evidence that the Python call sequence is bit-equivalent to
`ds_core/src/accounts/pow.rs`.

**API difference found:** wasmtime >= 40 changed `Memory.write` to
`write(store, value, start)` (value before offset). The Rust code's equivalent
`memory.write(store, ptr, bytes)` order will silently fail. Must use the new order.

## 2. TLS fingerprint + headers (`tools/spike_curl.py`) — PASS

All 17 identity headers land on the wire with the expected values.

**Two headers must be explicitly removed.** curl_cffi's `chrome136` profile
defaults to a top-level-navigation shape and adds:

- `sec-fetch-user: ?1`
- `upgrade-insecure-requests: 1`

The real DeepSeek frontend issues XHR/fetch calls, which never carry either.
Passing `None` as the header value removes the header entirely (verified: header
count dropped 14 -> 12). Do **not** pass `""` — that sends an empty header.

curl_cffi's default UA for `chrome136` is macOS Chrome 136; we always override it
with the Windows Chrome 140 UA, which is what the Rust client does.

Fingerprint archived to `docs/tls-fingerprint.json`:

```
ja3  2efc27707cba4eb26ed4c95e56df685e
ja4  t13d1516h2_8daaf6152771_d8a2da3f94cd
h2   1:65536;2:0;4:6291456;6:262144|15663105|0|m,a,s,p
```

## 3. Streaming + cancellation (`tools/spike_stream.py`) — PASS

- `aiter_content()` delivers frames incrementally (4 chunks for 4 frames).
- `CancelledError` raised inside the consumer propagates and the `finally` block
  runs, so `stop_stream` / `delete_session` cleanup is reliable.
- **`asyncio.wait_for(anext(gen), 1)` is unsafe.** On timeout the inner generator's
  `finally` never ran (leaked upstream connection). Confirmed hazard.
- **Correct pattern works:** hold one pending `anext` task and race it with
  `asyncio.wait({task}, timeout=1)`. The generator survives the timeout and can be
  resumed; we then drained the remaining 3 chunks normally.

This locks in the pipeline design: keepalive must use a persistent task, never
`wait_for`.

## Consequences for Phase 1

1. `core/pow.py` must use `memory.write(store, data, ptr)`.
2. `core/identity.py` must inject `Sec-Fetch-User: None` and
   `Upgrade-Insecure-Requests: None`.
3. `openai/response/pipeline.py` must implement the persistent-task keepalive.
4. Pin `wasmtime>=40` (API break) and `curl-cffi>=0.11` in pyproject.
