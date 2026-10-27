"""OpenAI request / response models.

``extra="allow"`` everywhere: the real API sends plenty of fields we do not act
on (temperature, top_p, max_tokens, stop, seed, logprobs...). Accepting and
ignoring them matches the Rust project, which never forwards them upstream.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageUrl(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str = ""
    detail: str | None = None


class FilePart(BaseModel):
    model_config = ConfigDict(extra="allow")

    filename: str | None = None
    file_data: str | None = None
    file_id: str | None = None


class InputAudio(BaseModel):
    model_config = ConfigDict(extra="allow")

    data: str = ""
    format: str = "wav"


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    image_url: ImageUrl | None = None
    file: FilePart | None = None
    input_audio: InputAudio | None = None
    refusal: str | None = None


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    type: str = "function"
    function: FunctionCall = Field(default_factory=FunctionCall)
    index: int | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: FunctionCall | None = None
    refusal: str | None = None


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        if self.description:
            out["description"] = self.description
        if self.schema_ is not None:
            out["schema"] = self.schema_
        if self.strict is not None:
            out["strict"] = self.strict
        return out


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "text"
    json_schema: JsonSchemaSpec | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[Message] = Field(default_factory=list)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    web_search_options: dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    user: str | None = None

    @property
    def wants_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = None


class ChatCompletionsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
