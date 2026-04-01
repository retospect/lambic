"""MCP client pool — connect to N stdio MCP servers, aggregate tools."""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from acatome_lambic.core.config import McpServer

log = logging.getLogger("lambic.mcp")


@dataclass
class RegisteredTool:
    """A tool from an MCP server, with enable/disable state."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    enabled: bool = True

    @property
    def qualified_name(self) -> str:
        return f"{self.server_name}.{self.name}"

    def to_schema(self) -> dict[str, Any]:
        """Convert to generic tool schema dict."""
        return {
            "name": self.qualified_name,
            "description": f"[{self.server_name}] {self.description}",
            "inputSchema": self.input_schema,
        }


@dataclass
class ServerConnection:
    """A live connection to an MCP server."""

    config: McpServer
    session: ClientSession
    tools: list[RegisteredTool] = field(default_factory=list)
    status: str = "connected"
    error: str = ""


class McpClientPool:
    """Manages connections to multiple MCP servers."""

    def __init__(self, servers: list[McpServer]) -> None:
        self._server_configs = servers
        self._connections: dict[str, ServerConnection] = {}
        self._tools: dict[str, RegisteredTool] = {}
        self._exit_stack = AsyncExitStack()

    @property
    def tools(self) -> dict[str, RegisteredTool]:
        """All registered tools, keyed by qualified name."""
        return self._tools

    @property
    def connections(self) -> dict[str, ServerConnection]:
        return self._connections

    def enabled_tool_schemas(self) -> list[dict[str, Any]]:
        """Get schemas for all enabled tools (for LLM)."""
        return [t.to_schema() for t in self._tools.values() if t.enabled]

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers sequentially.

        Sequential connection is required because stdio_client uses anyio
        task groups that bind to the creating task — asyncio.gather would
        enter contexts in child tasks, causing cancel-scope errors on close.
        """
        for s in self._server_configs:
            if s.enabled:
                await self._connect_server(s)

    async def _connect_server(self, server: McpServer) -> None:
        """Connect to a single MCP server via stdio."""
        try:
            params = StdioServerParameters(
                command=server.cmd[0],
                args=server.cmd[1:] if len(server.cmd) > 1 else [],
                env=server.env or None,
            )

            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            # List tools from this server
            result = await session.list_tools()
            conn = ServerConnection(config=server, session=session)

            for tool in result.tools:
                rt = RegisteredTool(
                    server_name=server.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=(
                        tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                    ),
                )
                conn.tools.append(rt)
                self._tools[rt.qualified_name] = rt

            self._connections[server.name] = conn
            log.info("Connected to %s (%d tools)", server.name, len(conn.tools))

        except Exception as exc:
            log.error("Failed to connect to %s: %s", server.name, exc)
            self._connections[server.name] = ServerConnection(
                config=server,
                session=None,  # type: ignore
                status="error",
                error=str(exc),
            )

    @staticmethod
    def _coerce_arguments(
        arguments: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Fix double-encoded JSON strings from LLMs.

        Some models (notably Claude) occasionally send complex nested
        parameters (arrays, objects) as JSON *strings* instead of actual
        JSON structures.  Use the tool's input schema to detect the
        mismatch and auto-parse.
        """
        props = schema.get("properties", {})
        if not props:
            return arguments

        fixed = dict(arguments)
        for key, val in fixed.items():
            if not isinstance(val, str):
                continue
            expected = props.get(key, {}).get("type")
            if expected in ("array", "object"):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, (list, dict)):
                        fixed[key] = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        return fixed

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool by qualified name, return result as string."""
        tool = self._tools.get(qualified_name)
        if not tool:
            return f"Error: unknown tool '{qualified_name}'"

        if not tool.enabled:
            return f"Error: tool '{qualified_name}' is disabled"

        conn = self._connections.get(tool.server_name)
        if not conn or conn.status != "connected":
            return f"Error: server '{tool.server_name}' not connected"

        arguments = self._coerce_arguments(arguments, tool.input_schema)

        try:
            result = await conn.session.call_tool(tool.name, arguments)
            # Flatten result content to string
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif hasattr(item, "data"):
                    parts.append(str(item.data))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        except Exception as exc:
            log.error("Tool call %s failed: %s", qualified_name, exc)
            return f"Error calling {qualified_name}: {exc}"

    def set_tools_enabled(self, pattern: str, enabled: bool) -> list[str]:
        """Enable/disable tools matching a pattern. Returns affected names.

        Matches against qualified name (server.tool), server name, or
        bare tool name.  If pattern has no glob chars, also tries
        pattern.* so bare server names work (e.g. 'perplexity').
        """
        import fnmatch

        patterns = [pattern]
        if not any(c in pattern for c in "*?[]"):
            patterns.append(f"{pattern}.*")

        affected = []
        for qname, tool in self._tools.items():
            candidates = (qname, tool.server_name, tool.name)
            if any(fnmatch.fnmatch(c, p) for c in candidates for p in patterns):
                tool.enabled = enabled
                affected.append(qname)
        return affected

    def tool_status(self) -> list[dict[str, Any]]:
        """Get status of all tools for display."""
        rows = []
        for name, tool in self._tools.items():
            conn = self._connections.get(tool.server_name)
            rows.append(
                {
                    "name": name,
                    "enabled": tool.enabled,
                    "server_status": conn.status if conn else "unknown",
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        return rows

    async def close(self) -> None:
        """Close all connections (best-effort, never raises)."""
        try:
            await self._exit_stack.aclose()
        except Exception as exc:
            log.debug("Error during MCP cleanup (ignored): %s", exc)
        self._connections.clear()
        self._tools.clear()
