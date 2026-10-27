"""Tool definitions and the system-prompt block that teaches the DSML dialect.

The upstream has no native function calling, so tools are described in the system
prompt and the model replies with a DSML block that ``response/tool_parser.py``
turns back into ``tool_calls``.

``legacy`` mode reproduces the Rust project's original Chinese instruction block
for the ``<|tool▁calls▁begin|>`` protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..types import ChatCompletionsRequest

DSML_EXAMPLE = """<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="city" string="true">Shanghai</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>"""


@dataclass(slots=True)
class ToolContext:
    """System-prompt fragments injected by :func:`build`."""

    protocol: str = "dsml"
    defs_text: str | None = None
    format_block: str | None = None
    instruction_text: str | None = None

    @property
    def has_tools(self) -> bool:
        return bool(self.defs_text)


def _qualified_name(function: dict[str, Any], tool: dict[str, Any]) -> tuple[str, str | None]:
    """Return ``(name_for_model, namespace)`` honouring all three namespace forms."""
    raw_name = str(function.get("name", ""))
    explicit = function.get("namespace") or tool.get("namespace")
    if isinstance(explicit, dict):
        explicit = explicit.get("name")
    if "::" in raw_name:
        namespace, bare = raw_name.split("::", 1)
        if explicit and explicit != namespace:
            from ...core.errors import BadRequest

            raise BadRequest(
                f"conflicting tool namespaces for {raw_name!r}: "
                f"name says {namespace!r}, field says {explicit!r}"
            )
        return f"{namespace}::{bare}", namespace
    if explicit:
        return f"{explicit}::{raw_name}", str(explicit)
    return raw_name, None


def _normalise_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type", "function") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        qualified, namespace = _qualified_name(function, tool)
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": qualified,
                "description": function.get("description", ""),
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
            },
        }
        if namespace:
            schema["function"]["namespace"] = namespace
        out.append(schema)
    return out


def _dsml_block(schemas: list[dict[str, Any]]) -> str:
    return (
        "## Tools\n\n"
        "You have access to the tools listed below. Call them only when the user's "
        "request genuinely requires them.\n\n"
        "### Tool call format\n\n"
        "Emit calls as a DSML block, exactly in this shape:\n\n"
        f"{DSML_EXAMPLE}\n\n"
        "Rules:\n"
        "1. Keep the space between `<｜DSML｜>` and the tag name. "
        "`<｜DSML｜ calls>` is correct; `<｜DSML｜calls>` is not.\n"
        '2. Every parameter must declare `string="true"` or `string="false"`.\n'
        '   - `string="true"`: the value is raw text, written as-is, with no quotes '
        "and no escaping.\n"
        '   - `string="false"`: the value is JSON (number, boolean, null, array, or '
        "object).\n"
        "3. Emit one `<｜DSML｜ invoke>` per call. Several invokes may share one "
        "`calls` block.\n"
        "4. Write any reasoning first, then the DSML block. Never put the DSML block "
        "inside a markdown code fence.\n"
        "5. Namespaced tools are written as `namespace::name`; `::` appears at most "
        "once.\n\n"
        "### Available Tool Schemas\n\n"
        "```json\n"
        f"{json.dumps(schemas, ensure_ascii=False, indent=2)}\n"
        "```"
    )


def _legacy_block(schemas: list[dict[str, Any]]) -> str:
    return (
        "## 可用工具\n\n"
        "你可以调用以下工具。需要时请在回复末尾输出工具调用块，格式为：\n\n"
        "<|tool▁calls▁begin|>\n"
        '[{"name": "工具名", "arguments": {"参数": "值"}}]\n'
        "<|tool▁calls▁end|>\n\n"
        "注意：arguments 必须是合法 JSON 对象；不要放在 markdown 代码块内。\n\n"
        "### 工具定义\n\n"
        "```json\n"
        f"{json.dumps(schemas, ensure_ascii=False, indent=2)}\n"
        "```"
    )


def _instruction(tool_choice: Any, parallel: bool | None) -> str | None:
    parts: list[str] = []
    if isinstance(tool_choice, dict):
        name = ""
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name", ""))
        elif isinstance(tool_choice.get("name"), str):
            name = tool_choice["name"]
        if name:
            parts.append(f"You must call the tool `{name}` in this turn.")
    if parallel is False:
        parts.append("Call at most one tool in this turn.")
    return "\n".join(parts) if parts else None


def extract(req: ChatCompletionsRequest, protocol: str = "dsml") -> ToolContext:
    """Build the system-prompt fragments for ``req.tools``."""
    ctx = ToolContext(protocol=protocol)
    if not req.tools:
        return ctx
    if isinstance(req.tool_choice, str) and req.tool_choice == "none":
        return ctx

    schemas = _normalise_tools(req.tools)
    if not schemas:
        return ctx

    block = _dsml_block(schemas) if protocol == "dsml" else _legacy_block(schemas)
    ctx.defs_text = block
    ctx.instruction_text = _instruction(req.tool_choice, req.parallel_tool_calls)
    return ctx
