"""``/v1/models`` payloads."""

from __future__ import annotations

import time

from ..settings import Config


def _entry(config: Config, model_id: str) -> dict:
    model_type = config.resolve_model_type(model_id)
    max_input, max_output, _chars = config.limits_for(model_type)
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "deepseek",
        "permission": [],
        "root": model_id,
        "parent": None,
        "context_window": max_input,
        "max_output_tokens": max_output,
    }


def list_models(config: Config) -> dict:
    return {
        "object": "list",
        "data": [_entry(config, model_id) for model_id in config.model_ids],
    }


def get_model(config: Config, model_id: str) -> dict | None:
    if model_id.lower() not in config.model_registry:
        return None
    return _entry(config, model_id)


def now() -> int:
    return int(time.time())
