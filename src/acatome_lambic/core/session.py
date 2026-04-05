"""Chat session — message history, tool dispatch, truncation, commands."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from acatome_lambic.core.config import ShellConfig
from acatome_lambic.core.llm import LlmClient, LlmResponse, ToolCall
from acatome_lambic.core.mcp_client import McpClientPool

log = logging.getLogger("lambic.session")

# Max tool-calling rounds per user turn (prevent infinite loops)
MAX_TOOL_ROUNDS = 100


def _build_tool_signature(name: str, schema: dict[str, Any]) -> str:
    """Build a function-signature string from tool name + JSON schema.

    Example: ``acatome.paper(id, filter="", page=1)``
    """
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = []
    for pname, pinfo in props.items():
        if pname in required:
            params.append(pname)
        else:
            # Show default if available, otherwise infer from type
            default = pinfo.get("default")
            if default is not None:
                params.append(f"{pname}={default!r}")
            else:
                ptype = pinfo.get("type", "")
                placeholder = {
                    "string": '""',
                    "integer": "0",
                    "number": "0",
                    "boolean": "false",
                    "array": "[]",
                    "object": "{}",
                }.get(ptype, "None")
                params.append(f"{pname}={placeholder}")
    return f"{name}({', '.join(params)})"


MAX_AUTOCONTINUE = 5

# Patterns that indicate the model tried to make a tool call via XML text
# instead of using the API's function calling mechanism.
_BROKEN_TOOL_PATTERNS = (
    "<function=",
    "<tool_call>",
    "<|tool_call|>",
    "</tool_call>",
    "<parameter=",
)


def _has_broken_tool_xml(text: str) -> bool:
    """Check if text contains XML-like tool call patterns."""
    return any(p in text for p in _BROKEN_TOOL_PATTERNS)


@dataclass
class ToolResult:
    """A completed tool call with its result."""

    call: ToolCall
    result: str
    full_result: str
    truncated: bool
    elapsed: float


@dataclass
class TurnEvent:
    """Events emitted during a turn, consumed by the TUI."""

    kind: str  # "text", "tool_call", "tool_result", "error", "thinking", "done"
    data: Any = None


class ChatSession:
    """Core chat session — no TUI dependency.

    Manages message history, LLM interaction, and tool dispatch.
    Can be driven by a TUI, a web backend, or tests.
    """

    def __init__(self, config: ShellConfig) -> None:
        self.config = config
        self.llm = LlmClient(config.llm)
        self.mcp = McpClientPool(config.servers)
        self.messages: list[dict[str, Any]] = []
        self.tool_results_full: dict[str, str] = {}
        self._tool_call_counter = 0
        self.autocontinue: bool = True
        self.task_reminder: str = ""  # injected on auto-continue

        if config.system_prompt:
            self.messages.append({"role": "system", "content": config.system_prompt})

    async def start(self) -> dict[str, Any]:
        """Initialize connections. Returns status dict."""
        ok = await self.llm.check_connection()
        if not ok:
            log.warning("LLM not reachable: %s", self.config.llm.spec)

        await self.mcp.connect_all()

        return {
            "llm_ok": ok,
            "model": self.config.llm.spec,
            "think": self.config.llm.think,
            "servers": {
                name: conn.status for name, conn in self.mcp.connections.items()
            },
            "tools": len(self.mcp.enabled_tool_schemas()),
        }

    def save_partial_response(self, content: str) -> None:
        """Save a partial assistant response to history (e.g. after Ctrl-C).

        If the last message is already an assistant message, do nothing
        (the turn completed normally before the interrupt was caught).
        """
        content = content.strip()
        if not content:
            return
        # Only add if the last message isn't already an assistant reply
        if self.messages and self.messages[-1].get("role") == "assistant":
            return
        self.messages.append(
            {"role": "assistant", "content": content + "\n\n[interrupted]"}
        )

    async def turn(self, user_input: str) -> AsyncIterator[TurnEvent]:
        """Process one user turn. Yields TurnEvents for the TUI."""
        # Check for message commands first (transform into LLM messages)
        if user_input.startswith("/") and self.config.message_commands:
            verb = user_input.strip().split()[0].lower().lstrip("/")
            if verb in self.config.message_commands:
                # Set task reminder if a reminder builder is registered
                if verb in self.config.task_reminder_commands:
                    self.task_reminder = self.config.task_reminder_commands[verb](
                        user_input
                    )
                user_input = self.config.message_commands[verb](user_input)
                # Fall through to normal LLM interaction

        # Check for slash commands
        if user_input.startswith("/"):
            result = self._handle_command(user_input)
            yield TurnEvent("command", result)
            yield TurnEvent("done")
            return

        self.messages.append({"role": "user", "content": user_input})

        tools = self.mcp.enabled_tool_schemas()
        autocontinue_count = 0

        for _round in range(MAX_TOOL_ROUNDS):
            log.info(
                "turn round=%d msgs=%d autocontinue=%s",
                _round,
                len(self.messages),
                self.autocontinue,
            )
            # Stream LLM response
            collected_content = ""
            final_response: LlmResponse | None = None

            _MAX_LLM_RETRIES = 3
            for _attempt in range(_MAX_LLM_RETRIES):
                try:
                    async for item in self.llm.stream(self.messages, tools or None):
                        if isinstance(item, str):
                            collected_content += item
                            yield TurnEvent("text", item)
                        elif isinstance(item, LlmResponse):
                            final_response = item
                    break  # success
                except Exception as exc:
                    is_rate_limit = (
                        "RateLimitError" in type(exc).__name__
                        or "rate_limit" in str(exc).lower()
                    )
                    if _attempt < _MAX_LLM_RETRIES - 1 and not is_rate_limit:
                        log.warning(
                            "LLM error (attempt %d/%d, retrying): %s",
                            _attempt + 1,
                            _MAX_LLM_RETRIES,
                            exc,
                        )
                        yield TurnEvent(
                            "status",
                            f"LLM error, retrying ({_attempt + 1}/{_MAX_LLM_RETRIES})...",
                        )
                        collected_content = ""
                        final_response = None
                        continue
                    log.error(
                        "LLM error (attempt %d/%d, giving up): %s",
                        _attempt + 1,
                        _MAX_LLM_RETRIES,
                        exc,
                    )
                    yield TurnEvent(
                        "error",
                        f"LLM error: {exc}\nIs {self.config.llm.spec} reachable?",
                    )
                    # Remove the user message we just added
                    if self.messages and self.messages[-1]["role"] == "user":
                        self.messages.pop()
                    yield TurnEvent("done")
                    return

            # If we got a final response with tool calls
            if final_response and final_response.tool_calls:
                if final_response.thinking:
                    yield TurnEvent("thinking", final_response.thinking)

                # Add assistant message with tool calls
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": final_response.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": (
                                        json.dumps(tc.arguments)
                                        if isinstance(tc.arguments, dict)
                                        else tc.arguments
                                    ),
                                },
                            }
                            for tc in final_response.tool_calls
                        ],
                    }
                )

                # Execute tool calls (parallel across servers)
                results = await self._execute_tools(final_response.tool_calls)
                has_error = False
                for tr in results:
                    yield TurnEvent("tool_result", tr)
                    if "ERROR" in tr.result[:20]:
                        has_error = True

                    # Add tool result to messages
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.call.id,
                            "content": tr.result,
                        }
                    )

                # Detect repeated error tool calls (model stuck)
                if has_error:
                    error_streak = 0
                    for msg in reversed(self.messages):
                        if msg["role"] == "tool" and "ERROR" in msg["content"][:20]:
                            error_streak += 1
                        elif msg["role"] == "tool" or msg["role"] == "user":
                            break
                    if error_streak >= 2:
                        # Try to find a concrete hint from recent tool results
                        hint = self._extract_last_hint()
                        nudge = (
                            "STOP calling tools with empty parameters. "
                            "You MUST pass arguments."
                        )
                        if hint:
                            nudge += f"\nDo exactly this: {hint}"
                        self.messages.append({"role": "user", "content": nudge})

                # Tool calls are productive — reset autocontinue counter
                autocontinue_count = 0
                # Continue loop — LLM will see tool results
                continue

            else:
                # No tool calls — final text response
                if final_response and final_response.thinking:
                    yield TurnEvent("thinking", final_response.thinking)

                content = (
                    final_response.content if final_response else collected_content
                )
                if content:
                    self.messages.append({"role": "assistant", "content": content})

                # Autocontinue decision
                truncated = (
                    final_response.stop_reason == "length" if final_response else False
                )
                thinking = (final_response.thinking or "") if final_response else ""
                has_tools = bool(tools)

                # Autocontinue when: truncated, or tools available (model
                # should be using them, not just writing text).
                should_autocontinue = (
                    self.autocontinue
                    and autocontinue_count < MAX_AUTOCONTINUE
                    and (truncated or has_tools)
                )

                stop_reason = (
                    final_response.stop_reason if final_response else "no_response"
                )
                log.info(
                    "round=%d stop=%s autocontinue=%s count=%d/%d tools=%s text=%d think=%d",
                    _round,
                    stop_reason,
                    should_autocontinue,
                    autocontinue_count,
                    MAX_AUTOCONTINUE,
                    has_tools,
                    len(collected_content),
                    len(thinking),
                )

                # Emit usage info (even for empty responses)
                yield TurnEvent(
                    "usage",
                    {
                        "prompt_tokens": (
                            final_response.prompt_tokens if final_response else 0
                        ),
                        "completion_tokens": (
                            final_response.completion_tokens if final_response else 0
                        ),
                        "stop_reason": stop_reason,
                        "max_tokens": self.config.llm.max_tokens,
                        "autocontinue": should_autocontinue,
                    },
                )

                # Auto-continue: inject continuation and loop
                if should_autocontinue:
                    autocontinue_count += 1
                    if truncated:
                        msg = "Continue from where you left off."
                    elif _has_broken_tool_xml(collected_content + thinking):
                        msg = (
                            "Your tool call was not recognized — it was "
                            "output as text instead of a function call. "
                            "Do NOT write XML tags. Just call the tool "
                            "normally."
                        )
                    elif not collected_content:
                        msg = (
                            "You produced no output or tool calls. "
                            "Continue and use the available tools."
                        )
                    else:
                        msg = "Continue. Use the available tools to do the work."
                    if self.task_reminder:
                        msg += f"\n\nReminder:\n{self.task_reminder}"
                    self.messages.append({"role": "user", "content": msg})
                    continue
                break
        else:
            # for-loop exhausted MAX_TOOL_ROUNDS without breaking
            log.warning("Hit MAX_TOOL_ROUNDS=%d — stopping turn", MAX_TOOL_ROUNDS)
            yield TurnEvent(
                "usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "stop_reason": f"max_rounds ({MAX_TOOL_ROUNDS})",
                    "max_tokens": self.config.llm.max_tokens,
                    "autocontinue": False,
                },
            )

        yield TurnEvent("done")

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls, parallel when possible."""

        async def _run_one(tc: ToolCall) -> ToolResult:
            self._tool_call_counter += 1
            call_id = f"tc_{self._tool_call_counter}"

            # Catch malformed JSON from the LLM (concatenated objects etc.)
            if "__parse_error__" in tc.arguments:
                return ToolResult(
                    call=tc,
                    result=(
                        "ERROR: Your tool call had malformed JSON arguments — "
                        "multiple JSON objects were concatenated. "
                        "Make ONE tool call at a time with valid arguments. "
                        "To read multiple chunks, use comma-separated ids: "
                        "get(id='slug1›4,slug2›9,slug3›3')"
                    ),
                    full_result=tc.arguments.get("__parse_error__", ""),
                    truncated=False,
                    elapsed=0,
                )

            start = time.monotonic()
            full_result = await self.mcp.call_tool(tc.name, tc.arguments)
            elapsed = time.monotonic() - start

            # Truncate if needed (head + tail so LLM sees both ends)
            truncated = False
            result = full_result
            max_len = self.config.max_tool_result
            if len(full_result) > max_len:
                head_len = int(max_len * 0.75)
                tail_len = max_len - head_len
                omitted = len(full_result) - head_len - tail_len
                result = (
                    full_result[:head_len] + f"\n\n[… {omitted} chars omitted. "
                    f"Use /expand {call_id} for full result]\n\n"
                    + full_result[-tail_len:]
                )
                truncated = True
                self.tool_results_full[call_id] = full_result

            return ToolResult(
                call=tc,
                result=result,
                full_result=full_result,
                truncated=truncated,
                elapsed=elapsed,
            )

        results = await asyncio.gather(
            *[_run_one(tc) for tc in tool_calls], return_exceptions=True
        )

        final = []
        for r in results:
            if isinstance(r, Exception):
                log.error("Tool execution error: %s", r)
                final.append(
                    ToolResult(
                        call=ToolCall(id="err", name="error", arguments={}),
                        result=f"Error: {r}",
                        full_result=str(r),
                        truncated=False,
                        elapsed=0,
                    )
                )
            else:
                final.append(r)
        return final

    # Regex to find concrete tool-call hints like get(id='slug›N')
    _HINT_RE = re.compile(r"(?:get|search)\((?:id|query)='[^']+?'[^)]*\)")

    def _extract_last_hint(self) -> str:
        """Scan recent tool results for a concrete hint the model can copy."""
        for msg in reversed(self.messages):
            if msg["role"] == "tool" and "ERROR" not in msg["content"][:20]:
                m = self._HINT_RE.search(msg["content"])
                if m:
                    return m.group(0)
            elif msg["role"] == "user":
                break
        return ""

    def _handle_command(self, cmd: str) -> str:
        """Handle slash commands. Returns response text."""
        parts = cmd.strip().split(None, 2)
        verb = parts[0].lower()

        if verb == "/model":
            if len(parts) < 2:
                return f"Current model: {self.config.llm.spec}"
            spec = parts[1]
            if "/" in spec:
                provider, model = spec.split("/", 1)
            else:
                provider, model = "ollama", spec
            self.config.llm.provider = provider
            self.config.llm.model = model
            self.llm = LlmClient(self.config.llm)
            return f"Switched to {self.config.llm.spec}"

        elif verb == "/think":
            if len(parts) < 2:
                return f"Think mode: {'on' if self.config.llm.think else 'off'}"
            on = parts[1].lower() in ("on", "true", "1", "yes")
            self.config.llm.think = on
            self.llm.config.think = on
            return f"Think mode: {'on' if on else 'off'}"

        elif verb == "/tools":
            if len(parts) == 1:
                rows = self.mcp.tool_status()
                if not rows:
                    return "No tools registered."
                lines = []
                for r in rows:
                    marker = "✓" if r["enabled"] else "✗"
                    srv = "●" if r["server_status"] == "connected" else "○"
                    sig = _build_tool_signature(r["name"], r.get("input_schema", {}))
                    desc = (r.get("description") or "").split("\n")[0].strip()
                    desc = desc.replace("[", "\\[")  # escape Rich markup
                    lines.append(f"{srv} {marker} {sig}")
                    if desc:
                        lines.append(f"    {desc}")
                return "\n".join(lines)
            action = parts[1].lower()
            rest = cmd.strip().split()[2:]  # all tokens after verb + action
            # Support both "/tools off perplexity grandmofty" and "/tools perplexity off"
            if action in ("on", "off"):
                patterns = rest if rest else ["*"]
                enabled = action == "on"
            elif rest and rest[-1].lower() in ("on", "off"):
                enabled = rest[-1].lower() == "on"
                patterns = [action] + list(rest[:-1])
            else:
                patterns = None
                enabled = False
            if patterns is not None:
                affected = []
                for pat in patterns:
                    affected.extend(self.mcp.set_tools_enabled(pat, enabled))
                label = "Enabled" if enabled else "Disabled"
                return f"{label} {len(affected)} tool(s): {', '.join(affected)}"
            # /tools <name> — show full details for a specific tool
            rows = self.mcp.tool_status()
            match = [r for r in rows if r["name"] == action]
            if not match:
                # Try partial match
                match = [r for r in rows if action in r["name"]]
            if match:
                r = match[0]
                sig = _build_tool_signature(r["name"], r.get("input_schema", {}))
                desc = (r.get("description") or "(no description)").replace("[", "\\[")
                return f"{sig}\n\n{desc}"
            return "Usage: /tools [on|off|<name>] [pattern]"

        elif verb == "/expand":
            if len(parts) < 2:
                return "Usage: /expand <call_id>"
            call_id = parts[1]
            full = self.tool_results_full.get(call_id)
            if not full:
                return f"No stored result for '{call_id}'"
            return full

        elif verb == "/status":
            think = "on" if self.config.llm.think else "off"
            ac = "on" if self.autocontinue else "off"
            lines = [
                f"  [green]●[/green] LLM: [bold]{self.config.llm.spec}[/bold] (think: {think})",
            ]
            for name, conn in self.mcp.connections.items():
                if conn.status == "connected":
                    icon = "[green]●[/green]"
                else:
                    icon = "[red]○[/red]"
                err = f" ({conn.error})" if conn.error else ""
                lines.append(f"  {icon} {name}: {conn.status}{err}")
            n_tools = len(self.mcp.enabled_tool_schemas())
            n_msgs = len(self.messages)
            lines.append(
                f"  [dim]{n_tools} tools, {n_msgs} messages, "
                f"max_tokens: {self.config.llm.max_tokens}, "
                f"autocontinue: {ac}[/dim]"
            )
            return "\n".join(lines)

        elif verb == "/prompt":
            tools = self.mcp.enabled_tool_schemas()
            lines = []
            for i, msg in enumerate(self.messages):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                tc = msg.get("tool_calls", [])
                lines.append(f"--- \\[{i}] {role} ---")
                if content:
                    lines.append(content.replace("[", "\\["))
                if tc:
                    for call in tc:
                        fn = call.get("function", {})
                        lines.append(
                            f"  → {fn.get('name', '?')}({fn.get('arguments', '')})"
                        )
                lines.append("")
            if tools:
                lines.append(f"--- tools ({len(tools)}) ---")
                for t in tools:
                    name = t.get("name", "?")
                    desc = t.get("description", "")
                    schema = t.get("inputSchema", {})
                    sig = _build_tool_signature(name, schema)
                    lines.append(f"\n  {sig}")
                    if desc:
                        for dline in desc.strip().split("\n"):
                            lines.append(f"    {dline.replace('[', chr(92) + '[')}")
                lines.append("")
            return "\n".join(lines)

        elif verb == "/clear":
            sys_msgs = [m for m in self.messages if m["role"] == "system"]
            self.messages = sys_msgs
            self.tool_results_full.clear()
            return "History cleared."

        elif verb in ("/quit", "/exit", "/q"):
            return "__QUIT__"

        elif verb == "/autocontinue":
            if len(parts) < 2:
                return f"autocontinue: {'on' if self.autocontinue else 'off'}"
            on = parts[1].lower() in ("on", "true", "1", "yes")
            self.autocontinue = on
            return f"autocontinue: {'on' if on else 'off'}"

        elif verb == "/max_tokens":
            if len(parts) < 2:
                return f"max_tokens: {self.config.llm.max_tokens}"
            try:
                val = int(parts[1])
            except ValueError:
                return "Usage: /max_tokens <number>"
            self.config.llm.max_tokens = val
            self.llm.config.max_tokens = val
            return f"max_tokens: {val}"

        elif verb == "/more":
            # Continue generation with optional token override
            extra_tokens = 0
            if len(parts) >= 2:
                try:
                    extra_tokens = int(parts[1])
                except ValueError:
                    return "Usage: /more [extra_tokens]"
            # Temporarily increase max_tokens if specified
            saved = self.config.llm.max_tokens
            if extra_tokens:
                self.config.llm.max_tokens = extra_tokens
                self.llm.config.max_tokens = extra_tokens
            self._more_restore_tokens = saved
            self._more_extra = extra_tokens
            return "__MORE__"

        elif verb in ("/help", "/?"):
            lines = [
                "```",
                "/model [provider/model]   switch LLM",
                "/think [on|off]           toggle reasoning mode",
                "/tools [on|off] [pattern] list/toggle tools",
                "/max_tokens [N]           show/set output token limit",
                "/more [N]                 continue generation (opt. N tokens)",
                "/autocontinue [on|off]    autocontinue on truncation",
                "/expand <call_id>         show full tool result",
                "/status                   session info",
                "/prompt                   dump full LLM prompt",
                "/clear                    clear history",
                "/quit                     exit",
            ]
            for cmd_name in sorted(self.config.message_commands):
                lines.append(f"/{cmd_name} <prompt>          → LLM")
            for cmd_name in sorted(self.config.custom_commands):
                lines.append(f"/{cmd_name}")
            lines.append("```")
            return "\n".join(lines)

        # App-specific custom commands
        bare = verb.lstrip("/")
        if bare in self.config.custom_commands:
            return self.config.custom_commands[bare](cmd)

        return f"Unknown command: {verb}. Type /help for commands."

    async def close(self) -> None:
        """Clean up connections."""
        await self.mcp.close()
        await self.llm.close()
