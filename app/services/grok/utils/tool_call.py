"""
Tool call utilities for OpenAI-compatible function calling.

Provides prompt-based emulation of tool calls by injecting tool definitions
into the system prompt and parsing structured responses.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def build_tool_prompt(
    tools: List[Dict[str, Any]],
    tool_choice: Optional[Any] = None,
    parallel_tool_calls: bool = True,
) -> str:
    """Generate a system prompt block describing available tools.

    Args:
        tools: List of OpenAI-format tool definitions.
        tool_choice: "auto", "required", "none", or {"type":"function","function":{"name":"..."}}.
        parallel_tool_calls: Whether multiple tool calls are allowed.

    Returns:
        System prompt string to prepend to the conversation.
    """
    if not tools:
        return ""

    # tool_choice="none" means don't mention tools at all
    if tool_choice == "none":
        return ""

    # Collect valid function tools info for building example
    tool_infos = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if name:
            tool_infos.append(func)

    # --- Section 1: Core constraint (front-loaded) ---
    lines = [
        "[SYSTEM TOOL CALLING PROTOCOL]",
        "",
    ]

    # Behavior section comes FIRST so the model sees it immediately
    if tool_choice == "required":
        lines.append("CONSTRAINT: You MUST call at least one tool below. Do NOT respond with text only.")
    elif isinstance(tool_choice, dict):
        func_info = tool_choice.get("function", {})
        forced_name = func_info.get("name", "")
        if forced_name:
            lines.append(f'CONSTRAINT: You MUST call the tool "{forced_name}" in your response.')
    else:
        # "auto" or default — this is the critical part
        lines.append("CONSTRAINT: You are an assistant equipped with tools. When the user's request relates to ANY capability provided by the tools below, you MUST invoke the tool — do NOT answer from your own knowledge. You have NO internal access to real-time data, external APIs, or any information the tools can provide. Answering without calling a relevant tool is a protocol violation.")
        lines.append("")
        lines.append("Only respond with plain text if NONE of the listed tools are relevant.")

    lines.append("")

    # --- Section 2: Format ---
    lines.append("FORMAT: To call a tool, output a <tool_call> block with valid JSON:")
    lines.append("")
    lines.append("<tool_call>")
    lines.append('{"name": "tool_name", "arguments": {"key": "value"}}')
    lines.append("</tool_call>")
    lines.append("")
    lines.append("- JSON must be valid. No markdown fences. No extra wrapping.")
    lines.append("- You may include brief text before the <tool_call> block.")

    if parallel_tool_calls:
        lines.append("- You may use multiple <tool_call> blocks for multiple calls.")
    lines.append("")

    # --- Section 3: Tool definitions ---
    lines.append("TOOLS:")
    lines.append("")
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        if params:
            lines.append(f"  Parameters: {json.dumps(params, ensure_ascii=False)}")
        lines.append("")

    # --- Section 4: Example using first tool ---
    if tool_infos:
        first = tool_infos[0]
        ex_name = first.get("name", "tool")
        ex_params = first.get("parameters", {})
        ex_props = ex_params.get("properties", {})
        # Build a simple example arguments dict
        ex_args = {}
        for pname, pinfo in list(ex_props.items())[:2]:
            ex_args[pname] = f"<{pinfo.get('description', pname)}>"
        ex_args_str = json.dumps(ex_args, ensure_ascii=False)

        lines.append("EXAMPLE:")
        lines.append(f'User: (a request relevant to {ex_name})')
        lines.append("Assistant:")
        lines.append("<tool_call>")
        lines.append(f'{{"name": "{ex_name}", "arguments": {ex_args_str}}}')
        lines.append("</tool_call>")
        lines.append("")

    lines.append("[END TOOL CALLING PROTOCOL]")

    return "\n".join(lines)


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    if not text:
        return text
    start = text.find("{")
    if start == -1:
        return text
    end = text.rfind("}")
    if end == -1:
        return text[start:]
    if end < start:
        return text
    return text[start : end + 1]


def _remove_trailing_commas(text: str) -> str:
    if not text:
        return text
    return re.sub(r",\s*([}\]])", r"\1", text)


def _balance_braces(text: str) -> str:
    if not text:
        return text
    open_count = 0
    close_count = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_count += 1
        elif ch == "}":
            close_count += 1
    if open_count > close_count:
        text = text + ("}" * (open_count - close_count))
    return text


def _repair_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    cleaned = _extract_json_object(cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", " ")
    cleaned = _remove_trailing_commas(cleaned)
    cleaned = _balance_braces(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def parse_tool_call_block(
    raw_json: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not raw_json:
        return None
    parsed = None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = _repair_json(raw_json)
    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name")
    arguments = parsed.get("arguments", {})
    if not name:
        return None

    valid_names = set()
    if tools:
        for tool in tools:
            func = tool.get("function", {})
            tool_name = func.get("name")
            if tool_name:
                valid_names.add(tool_name)
    if valid_names and name not in valid_names:
        return None

    if isinstance(arguments, dict):
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, str):
        arguments_str = arguments
    else:
        arguments_str = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments_str},
    }


def parse_tool_calls(
    content: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """Parse tool call blocks from model output.

    Detects ``<tool_call>...</tool_call>`` blocks, parses JSON from each block,
    and returns OpenAI-format tool call objects.

    Args:
        content: Raw model output text.
        tools: Optional list of tool definitions for name validation.

    Returns:
        Tuple of (text_content, tool_calls_list).
        - text_content: text outside <tool_call> blocks (None if empty).
        - tool_calls_list: list of OpenAI tool call dicts, or None if no calls found.
    """
    if not content:
        return content, None

    matches = list(_TOOL_CALL_RE.finditer(content))
    if not matches:
        return content, None

    tool_calls = []
    for match in matches:
        raw_json = match.group(1).strip()
        tool_call = parse_tool_call_block(raw_json, tools)
        if tool_call:
            tool_calls.append(tool_call)

    if not tool_calls:
        return content, None

    # Extract text outside of tool_call blocks
    text_parts = []
    last_end = 0
    for match in matches:
        before = content[last_end:match.start()]
        if before.strip():
            text_parts.append(before.strip())
        last_end = match.end()
    trailing = content[last_end:]
    if trailing.strip():
        text_parts.append(trailing.strip())

    text_content = "\n".join(text_parts) if text_parts else None

    return text_content, tool_calls


def format_tool_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert assistant messages with tool_calls and tool role messages into text format.

    Since Grok's web API only accepts a single message string, this converts
    tool-related messages back to a text representation for multi-turn conversations.

    Args:
        messages: List of OpenAI-format messages that may contain tool_calls and tool roles.

    Returns:
        List of messages with tool content converted to text format.
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")

        if role == "assistant" and tool_calls:
            # Convert assistant tool_calls to text representation
            parts = []
            if content:
                parts.append(content if isinstance(content, str) else str(content))
            for tc in tool_calls:
                func = tc.get("function", {})
                tc_name = func.get("name", "")
                tc_args = func.get("arguments", "{}")
                tc_id = tc.get("id", "")
                parts.append(f'<tool_call>{{"name":"{tc_name}","arguments":{tc_args}}}</tool_call>')
            result.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        elif role == "tool":
            # Convert tool result to text format
            tool_name = name or "unknown"
            call_id = tool_call_id or ""
            content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False) if content else ""
            result.append({
                "role": "user",
                "content": f"tool ({tool_name}, {call_id}): {content_str}",
            })

        else:
            result.append(msg)

    return result


__all__ = [
    "build_tool_prompt",
    "parse_tool_calls",
    "format_tool_history",
    "parse_tool_call_block",
]
