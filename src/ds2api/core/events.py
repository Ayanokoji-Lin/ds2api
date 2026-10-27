"""Stream event model shared by the core and the OpenAI adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Meta:
    """First event: which account served the request (used for the x-ds-account header)."""

    account_id: str


@dataclass(slots=True, frozen=True)
class ThinkStart:
    pass


@dataclass(slots=True, frozen=True)
class ThinkDelta:
    content: str


@dataclass(slots=True, frozen=True)
class ContentStart:
    pass


@dataclass(slots=True, frozen=True)
class ContentDelta:
    content: str


@dataclass(slots=True, frozen=True)
class Done:
    finish_reason: str | None = None
    accumulated_token_usage: int | None = None


StreamEvent = Meta | ThinkStart | ThinkDelta | ContentStart | ContentDelta | Done
