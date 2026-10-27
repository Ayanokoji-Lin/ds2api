"""Prompt construction: OpenAI messages -> DeepSeek ChatML text.

Port of ``src/openai_adapter/request/prompt.rs``. Byte-exact reproduction matters:
the upstream only ever sees one flat ``prompt`` string, so every separator here is
part of the wire protocol.

Symbols are deliberately mixed:

* role tags use full-width bars: ``<｜User｜>``
* the legacy tool protocol uses ASCII bars + U+2581: ``<|tool▁calls▁begin|>``
"""

from __future__ import annotations

import json

from .tools import ToolContext
from ..types import ChatCompletionsRequest, ContentPart, Message
from ..response.tags import (
    EOS,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_OUTPUT_BEGIN,
    TOOL_OUTPUT_END,
    TOOL_OUTPUTS_BEGIN,
    TOOL_OUTPUTS_END,
)

MessageContent = str | list[ContentPart]


def merge_messages(messages: list[Message]) -> list[Message]:
    """Merge consecutive messages with the same role.

    ``tool`` messages are excluded: they are grouped later, and merging them here
    would lose the per-call boundaries.
    """
    merged: list[Message] = []
    for msg in messages:
        last = merged[-1] if merged else None
        if last is not None and last.role == msg.role and msg.role != "tool":
            if msg.content is not None:
                if last.content is None:
                    last.content = msg.content
                elif isinstance(last.content, str) and isinstance(msg.content, str):
                    last.content = f"{last.content}\n{msg.content}"
                elif isinstance(last.content, list) and isinstance(msg.content, list):
                    last.content = [*last.content, *msg.content]
                else:
                    left = format_content(last.content)
                    right = format_content(msg.content)
                    last.content = f"{left}\n{right}"
            if msg.tool_calls:
                last.tool_calls = [*(last.tool_calls or []), *msg.tool_calls]
            if msg.name is not None:
                last.name = msg.name
            if msg.tool_call_id is not None:
                last.tool_call_id = msg.tool_call_id
            if msg.function_call is not None:
                last.function_call = msg.function_call
            if msg.refusal is not None:
                last.refusal = msg.refusal
            continue
        merged.append(msg.model_copy(deep=True))
    return merged


def role_tag(role: str) -> str:
    if not role:
        return "<｜｜>"
    return f"<｜{role[0].upper()}{role[1:]}｜>"


def format_content(content: MessageContent | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Parts are joined with a newline (not an empty string).
    return "\n".join(format_part(part) for part in content)


def format_part(part: ContentPart) -> str:
    kind = part.type
    if kind == "text":
        return part.text or ""
    if kind == "refusal":
        return part.refusal or ""
    if kind == "image_url":
        if part.image_url is None:
            return "[图片]"
        url = part.image_url.url
        if url.startswith("http://") or url.startswith("https://"):
            return f"[请访问这个链接: {url}]"
        detail = part.image_url.detail or "auto"
        return f"[图片: detail={detail}]"
    if kind == "input_audio":
        fmt = part.input_audio.format if part.input_audio else "unknown"
        return f"[音频: format={fmt}]"
    if kind == "file":
        filename = (part.file.filename if part.file else None) or "unknown"
        desc = part.text or ""
        if desc:
            return f"[文件: {desc} (filename={filename})]"
        return f"[文件: filename={filename}]"
    return f"[未支持的内容类型: {kind}]"


def _json_or_null(raw: str) -> str:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = None
    return json.dumps(value, ensure_ascii=False)


def _format_tool_call_items(msg: Message) -> list[str]:
    items: list[str] = []
    for call in msg.tool_calls or []:
        fn = call.function
        items.append(
            '{"name": %s, "arguments": %s}'
            % (json.dumps(fn.name, ensure_ascii=False), _json_or_null(fn.arguments))
        )
    if msg.function_call is not None:
        fn = msg.function_call
        items.append(
            '{"name": %s, "arguments": %s}'
            % (json.dumps(fn.name, ensure_ascii=False), _json_or_null(fn.arguments))
        )
    return items


def format_assistant(msg: Message) -> str:
    parts: list[str] = []
    if msg.content is not None:
        parts.append(format_content(msg.content))
    items = _format_tool_call_items(msg)
    if items:
        parts.append(f"{TOOL_CALL_START}\n[{', '.join(items)}]\n{TOOL_CALL_END}")
    if msg.refusal:
        parts.append(f"(refusal: {msg.refusal})")
    return "\n".join(parts)


def format_generic(msg: Message) -> str:
    parts: list[str] = []
    if msg.name:
        parts.append(f"(name: {msg.name})")
    if msg.content is not None:
        parts.append(format_content(msg.content))
    return "\n".join(parts)


def format_tool_legacy(msg: Message) -> str:
    content = format_content(msg.content)
    return (
        f"{TOOL_OUTPUTS_BEGIN}{TOOL_OUTPUT_BEGIN}{content}"
        f"{TOOL_OUTPUT_END}{TOOL_OUTPUTS_END}"
    )


def format_message(msg: Message, *, protocol: str = "dsml") -> str:
    if msg.role == "assistant":
        body = format_assistant(msg)
    elif msg.role == "tool":
        body = format_tool_legacy(msg)
    elif msg.role == "function":
        body = format_generic(msg)
    else:
        body = format_generic(msg)

    tag = "" if msg.role == "tool" else role_tag(msg.role)
    prefix = EOS if msg.role == "user" else ""
    return f"{prefix}{tag}{body}"


def format_response_text(req: ChatCompletionsRequest) -> str:
    rf = req.response_format
    if rf is None:
        return ""
    kind = rf.type
    if kind == "json_object":
        return "请直接输出合法的 JSON 对象，不要包含任何 markdown 代码块标记或其他解释性文字。"
    if kind == "json_schema":
        schema = rf.json_schema.as_dict() if rf.json_schema else {}
        if not schema:
            return "以 JSON 的形式输出。"
        return f"以 JSON 的形式输出，输出的 JSON 需遵守以下的格式：\n\n~~~json\n{json.dumps(schema, ensure_ascii=False)}\n~~~"
    if kind == "text":
        return ""
    return f"请以 {kind} 格式输出。"


def build(req: ChatCompletionsRequest, tool_ctx: ToolContext) -> str:
    protocol = tool_ctx.protocol
    messages = merge_messages(req.messages)
    parts: list[str] = []

    index = 0
    while index < len(messages):
        msg = messages[index]
        if msg.role == "tool":
            contents: list[str] = []
            while index < len(messages) and messages[index].role == "tool":
                if messages[index].content is not None:
                    contents.append(format_content(messages[index].content))
                index += 1
            if protocol == "dsml":
                # DSML has no `tool` role: results ride in the next user turn.
                inner = "".join(f"<tool_result>{c}</tool_result>" for c in contents)
                parts.append(f"{EOS}{ROLE_USER}{inner}")
            else:
                inner = "".join(
                    f"{TOOL_OUTPUT_BEGIN}{c}{TOOL_OUTPUT_END}" for c in contents
                )
                parts.append(f"{TOOL_OUTPUTS_BEGIN}{inner}{TOOL_OUTPUTS_END}")
            continue
        parts.append(format_message(msg, protocol=protocol))
        index += 1

    system_sections: list[str] = []
    for text in (tool_ctx.defs_text, tool_ctx.format_block, tool_ctx.instruction_text):
        if text:
            system_sections.append(text)
    response_text = format_response_text(req)
    if response_text:
        system_sections.append(response_text)

    if system_sections:
        body = "\n\n".join(system_sections)
        target = next((p for p in parts if p.startswith(ROLE_SYSTEM)), None)
        if target is not None:
            insert_at = target.rfind("\n")
            if insert_at < 0:
                insert_at = len(target)
            position = parts.index(target)
            parts[position] = target[:insert_at] + f"\n\n{body}" + target[insert_at:]
        else:
            parts.insert(0, f"{ROLE_SYSTEM}{body}\n")

    # The generation anchor must be judged on the LAST part, not "any part".
    if not parts or not parts[-1].startswith(ROLE_ASSISTANT):
        parts.append(f"{ROLE_ASSISTANT}\n")

    return "".join(parts)
