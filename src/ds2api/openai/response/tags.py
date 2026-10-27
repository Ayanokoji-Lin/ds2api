"""Tag constants shared by prompt building and response parsing.

Two different tag families coexist and must never be mixed:

* **ChatML role tags** use full-width vertical bars (U+FF5C):
  ``<｜System｜>``, ``<｜User｜>``, ``<｜Assistant｜>``, ``<｜end▁of▁sentence｜>``.
* **Legacy tool tags** use ASCII pipes plus U+2581:
  ``<|tool▁calls▁begin|>``.
* **DSML tags** use U+FF5C plus a required space before the name (V4.1):
  ``<｜DSML｜ calls>``.

The space rule changed between V4 (``<｜DSML｜tool_calls>``) and V4.1
(``<｜DSML｜ calls>``), and parsers must tolerate both plus the ASCII variants
seen in the wild (``<||DSML||tool_calls>``, ``<｜｜DSML｜｜tool_calls>``).
"""

from __future__ import annotations

import re

# -- ChatML role tags ------------------------------------------------------
ROLE_SYSTEM = "<｜System｜>"
ROLE_USER = "<｜User｜>"
ROLE_ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"

# -- legacy tool tags (ASCII pipe + U+2581) --------------------------------
TOOL_CALL_START = "<|tool▁calls▁begin|>"
TOOL_CALL_END = "<|tool▁calls▁end|>"
TOOL_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"
TOOL_OUTPUTS_END = "<｜tool▁outputs▁end｜>"
TOOL_OUTPUT_BEGIN = "<｜tool▁output▁begin｜>"
TOOL_OUTPUT_END = "<｜tool▁output▁end｜>"

# -- thinking tags ---------------------------------------------------------
THINK_OPEN = "<｜think｜>"
THINK_CLOSE = "<｜/think｜>"

# -- DSML ------------------------------------------------------------------
#: Matches an opening DSML tag in any known spelling and captures the tag name
#: plus the raw attribute blob.
DSML_OPEN_RE = re.compile(
    r"<\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*([A-Za-z_][A-Za-z0-9_]*)\b([^<>]*)>"
)
#: Matches a closing DSML tag.
DSML_CLOSE_RE = re.compile(
    r"<\s*/\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*([A-Za-z_][A-Za-z0-9_]*)\s*>"
)
DSML_ATTR_RE = re.compile(r"""([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*("([^"]*)"|'([^']*)')""")

DSML_TAG_CALLS = "calls"
DSML_TAG_TOOL_CALLS = "tool_calls"
DSML_TAG_INVOKE = "invoke"
DSML_TAG_PARAMETER = "parameter"

DSML_CALLS_NAMES = frozenset({DSML_TAG_CALLS, DSML_TAG_TOOL_CALLS})

#: Sliding-window size for streamed tag detection, in bytes.
#:
#: The Rust project uses 71, sized for ``<|tool▁calls▁begin|>``. DSML's longest
#: closing tag is ``</｜DSML｜ parameter>`` which is ~30 bytes in UTF-8 (each
#: full-width bar is 3 bytes), so 128 leaves ample headroom.
TAG_WINDOW = 128

#: Legacy window, kept for the legacy parser path.
TAG_WINDOW_LEGACY = 71

#: Hard cap on buffered tool-call XML before falling back to plain text.
MAX_XML_BUF_LEN = 64 * 1024

_FULLWIDTH_BAR = "\uff5c"
_LOWER_ONE_EIGHTH = "\u2581"


def norm_tag_char(ch: str) -> str:
    """Fold look-alike characters so fuzzy tag matching works."""
    if ch == _FULLWIDTH_BAR:
        return "|"
    if ch == _LOWER_ONE_EIGHTH:
        return "_"
    return ch


def normalize_tags(text: str) -> str:
    return "".join(norm_tag_char(c) for c in text)


def is_inside_code_fence(text: str, pos: int) -> bool:
    """True when ``pos`` sits inside a triple-backtick block."""
    return text.count("```", 0, pos) % 2 == 1


def split_qualified_name(name: str) -> tuple[str | None, str]:
    """Split ``ns::tool`` into ``("ns", "tool")``; raise on malformed names."""
    if "::" not in name:
        return None, name
    parts = name.split("::")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        from ...core.errors import BadRequest

        raise BadRequest(f"invalid qualified tool name: {name!r} (only one '::' allowed)")
    return parts[0], parts[1]
