"""Request statistics with optional JSON persistence.

Ported subset of ``src/server/stats.rs``: counters, per-model breakdown, and a
200-entry ring buffer of recent requests. Persisted every 30 requests and on
graceful shutdown.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PERSIST_INTERVAL = 30
LOG_CAPACITY = 200


@dataclass(slots=True)
class Stats:
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: int = 0
    model_stats: dict[str, dict] = field(default_factory=dict)
    request_logs: deque = field(default_factory=lambda: deque(maxlen=LOG_CAPACITY))
    _since_persist: int = 0
    _path: Path | None = None

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> Stats:
        stats = cls(_path=path)
        if not path.exists():
            return stats
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return stats
        stats.total_requests = int(raw.get("total_requests", 0))
        stats.success_requests = int(raw.get("success_requests", 0))
        stats.failed_requests = int(raw.get("failed_requests", 0))
        stats.total_prompt_tokens = int(raw.get("total_prompt_tokens", 0))
        stats.total_completion_tokens = int(raw.get("total_completion_tokens", 0))
        stats.model_stats = dict(raw.get("model_stats", {}))
        for entry in raw.get("request_logs", []):
            stats.request_logs.append(entry)
        return stats

    def persist_now(self) -> None:
        if self._path is None:
            return
        snapshot = self.snapshot(include_logs=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        self._since_persist = 0

    # -- recording ---------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool,
        request_id: str = "",
        api_key: str = "",
    ) -> None:
        self.total_requests += 1
        if success:
            self.success_requests += 1
        else:
            self.failed_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency_ms += latency_ms

        bucket = self.model_stats.setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        )
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["requests"] += 1

        self.request_logs.appendleft(
            {
                "timestamp": int(time.time()),
                "request_id": request_id,
                "model": model,
                "api_key": (api_key[:8] + "***") if api_key else "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "success": success,
            }
        )

        self._since_persist += 1
        if self._since_persist >= PERSIST_INTERVAL:
            self.persist_now()

    # -- views -------------------------------------------------------------

    def snapshot(self, *, include_logs: bool = False) -> dict:
        total = self.total_requests or 0
        data = {
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_latency_ms": int(self.total_latency_ms / total) if total else 0,
            "model_stats": self.model_stats,
        }
        if include_logs:
            data["request_logs"] = list(self.request_logs)
        return data

    def recent_logs(self, limit: int = 50) -> list[dict]:
        return list(self.request_logs)[: max(0, limit)]
