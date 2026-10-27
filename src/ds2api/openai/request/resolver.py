"""Model / thinking / search resolution.

Port of ``request/resolver.rs`` plus the ``has_http_urls`` rule from
``openai_adapter.rs``.

Note: the Rust source has a comment claiming ``default_search_enabled`` defaults
to true. It does not - ``config.rs`` returns false. We follow the code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...settings import Config
from ..types import ChatCompletionsRequest


@dataclass(slots=True)
class Resolved:
    model_type: str
    thinking_enabled: bool
    search_enabled: bool


def resolve(config: Config, req: ChatCompletionsRequest, *, has_http_urls: bool = False) -> Resolved:
    model_type = config.resolve_model_type(req.model)

    # Any effort except the literal "none" enables thinking; the default is "high".
    effort = req.reasoning_effort or "high"
    thinking_enabled = effort != "none"

    search_enabled = (
        req.web_search_options is not None
        or config.ds_core.default_search_enabled
        or has_http_urls
    )

    return Resolved(
        model_type=model_type,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
    )
