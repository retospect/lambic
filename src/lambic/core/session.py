"""Chat session — message history, tool dispatch, truncation, commands."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from lambic.core.config import LlmConfig, ShellConfig
from lambic.core.llm import LlmClient, LlmResponse, ToolCall
from lambic.core.mcp_client import McpClientPool

log = logging.getLogger("lambic.session")

# Max tool-calling rounds per user turn (prevent infinite loops)
MAX_TOOL_ROUNDS = 10
MAX_AUTOCONTINUE = 5


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
        self.autocontinue: bool = False

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

    async def turn(self, user_input: str) -> AsyncIterator[TurnEvent]:
        """Process one user turn. Yields TurnEvents for the TUI."""
        # Check for slash commands
        if user_input.startswith("/"):
            result = self._handle_command(user_input)
            yield TurnEvent("text", result)
            yield TurnEvent("done")
            return

        self.messages.append({"role": "user", "content": user_input})

        tools = self.mcp.enabled_tool_schemas()
        autocontinue_count = 0

        for _round in range(MAX_TOOL_ROUNDS):
            # Stream LLM response
            collected_content = ""
            final_response: LlmResponse | None = None

            try:
                async for item in self.llm.stream(self.messages, tools or None):
                    if isinstance(item, str):
                        collected_content += item
                        yield TurnEvent("text", item)
                    elif isinstance(item, LlmResponse):
                        final_response = item
            except Exception as exc:
                log.error("LLM error: %s", exc)
                yield TurnEvent(
                    "error",
                    f"LLM error: {exc}\n"
                    f"Is {self.config.llm.spec} reachable?",
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
                                    "arguments": json.dumps(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else tc.arguments,
                                },
                            }
                            for tc in final_response.tool_calls
                        ],
                    }
                )

                # Execute tool calls (parallel across servers)
                results = await self._execute_tools(final_response.tool_calls)
                for tr in results:
                    yield TurnEvent("tool_result", tr)

                    # Add tool result to messages
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.call.id,
                            "content": tr.result,
                        }
                    )

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

                # Emit usage info
                if final_response:
                    truncated = final_response.stop_reason == "length"
                    # Autocontinue: also trigger when model has tools but
                    # produced only text (described a plan but didn't act)
                    should_autocontinue = False
                    if self.autocontinue and autocontinue_count < MAX_AUTOCONTINUE:
                        if truncated:
                            should_autocontinue = True
                        elif tools and content:
                            # Model had tools available but chose to output
                            # text instead — nudge it to act
                            should_autocontinue = True

                    log.info(
                        "stop_reason=%s autocontinue=%s count=%d/%d",
                        final_response.stop_reason,
                        should_autocontinue,
                        autocontinue_count,
                        MAX_AUTOCONTINUE,
                    )

                    yield TurnEvent(
                        "usage",
                        {
                            "prompt_tokens": final_response.prompt_tokens,
                            "completion_tokens": final_response.completion_tokens,
                            "stop_reason": final_response.stop_reason,
                            "max_tokens": self.config.llm.max_tokens,
                            "autocontinue": should_autocontinue,
                        },
                    )

                    # Auto-continue: inject continuation and loop
                    if should_autocontinue:
                        autocontinue_count += 1
                        if truncated:
                            msg = "Continue from where you left off."
                        else:
                            msg = "Continue. Use the available tools to do the work."
                        self.messages.append({"role": "user", "content": msg})
                        continue
                break

        yield TurnEvent("done")

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls, parallel when possible."""

        async def _run_one(tc: ToolCall) -> ToolResult:
            self._tool_call_counter += 1
            call_id = f"tc_{self._tool_call_counter}"

            start = time.monotonic()
            full_result = await self.mcp.call_tool(tc.name, tc.arguments)
            elapsed = time.monotonic() - start

            # Truncate if needed
            truncated = False
            result = full_result
            max_len = self.config.max_tool_result
            if len(full_result) > max_len:
                result = (
                    full_result[:max_len]
                    + f"\n[truncated {len(full_result)} → {max_len} chars. "
                    f"Use /expand {call_id} for full result]"
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
                    lines.append(f"  {srv} {marker} {r['name']:40s} {r['description']}")
                return "\n".join(lines)
            action = parts[1].lower()
            pattern = parts[2] if len(parts) > 2 else "*"
            if action == "off":
                affected = self.mcp.set_tools_enabled(pattern, False)
                return f"Disabled {len(affected)} tool(s): {', '.join(affected)}"
            elif action == "on":
                affected = self.mcp.set_tools_enabled(pattern, True)
                return f"Enabled {len(affected)} tool(s): {', '.join(affected)}"
            return "Usage: /tools [on|off] [pattern]"

        elif verb == "/expand":
            if len(parts) < 2:
                return "Usage: /expand <call_id>"
            call_id = parts[1]
            full = self.tool_results_full.get(call_id)
            if not full:
                return f"No stored result for '{call_id}'"
            return full

        elif verb == "/status":
            lines = [
                f"Model: {self.config.llm.spec}",
                f"Think: {'on' if self.config.llm.think else 'off'}",
                f"Max tokens: {self.config.llm.max_tokens}",
                f"Autocontinue: {'on' if self.autocontinue else 'off'}",
                f"Messages: {len(self.messages)}",
                f"Tools: {len(self.mcp.enabled_tool_schemas())} enabled",
            ]
            for name, conn in self.mcp.connections.items():
                status_icon = "●" if conn.status == "connected" else "○"
                err = f" ({conn.error})" if conn.error else ""
                lines.append(f"  {status_icon} {name}: {conn.status}{err}")
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
            return (
                "```\n"
                "/model [provider/model]   switch LLM\n"
                "/think [on|off]           toggle reasoning mode\n"
                "/tools [on|off] [pattern] list/toggle tools\n"
                "/max_tokens [N]           show/set output token limit\n"
                "/more [N]                 continue generation (opt. N tokens)\n"
                "/autocontinue [on|off]    autocontinue on truncation\n"
                "/expand <call_id>         show full tool result\n"
                "/status                   session info\n"
                "/clear                    clear history\n"
                "/quit                     exit\n"
                "```"
            )

        return f"Unknown command: {verb}. Type /help for commands."

    async def close(self) -> None:
        """Clean up connections."""
        await self.mcp.close()
        await self.llm.close()
