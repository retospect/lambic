"""Tests for MCP client pool — unit tests (no actual MCP servers)."""

from __future__ import annotations

import pytest

from lambic.core.mcp_client import McpClientPool, RegisteredTool


class TestRegisteredTool:
    def test_qualified_name(self):
        t = RegisteredTool(
            server_name="acatome",
            name="paper",
            description="Read a paper",
            input_schema={"type": "object"},
        )
        assert t.qualified_name == "acatome.paper"

    def test_to_schema(self):
        t = RegisteredTool(
            server_name="acatome",
            name="paper",
            description="Read a paper",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        )
        schema = t.to_schema()
        assert schema["name"] == "acatome.paper"
        assert "[acatome]" in schema["description"]
        assert "properties" in schema["inputSchema"]

    def test_enabled_default(self):
        t = RegisteredTool(server_name="s", name="t", description="", input_schema={})
        assert t.enabled is True


class TestMcpClientPool:
    def test_empty_pool(self):
        pool = McpClientPool([])
        assert pool.enabled_tool_schemas() == []
        assert pool.tool_status() == []

    def test_set_tools_enabled(self):
        pool = McpClientPool([])
        # Manually register tools for testing
        pool._tools["acatome.paper"] = RegisteredTool(
            server_name="acatome",
            name="paper",
            description="Read",
            input_schema={},
        )
        pool._tools["acatome.search"] = RegisteredTool(
            server_name="acatome",
            name="search",
            description="Search",
            input_schema={},
        )
        pool._tools["precis.toc"] = RegisteredTool(
            server_name="precis",
            name="toc",
            description="TOC",
            input_schema={},
        )

        # Disable all acatome tools
        affected = pool.set_tools_enabled("acatome.*", False)
        assert len(affected) == 2
        assert not pool._tools["acatome.paper"].enabled
        assert not pool._tools["acatome.search"].enabled
        assert pool._tools["precis.toc"].enabled

        # Only precis tools in enabled schemas
        schemas = pool.enabled_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "precis.toc"

        # Re-enable
        affected = pool.set_tools_enabled("acatome.*", True)
        assert len(affected) == 2
        assert len(pool.enabled_tool_schemas()) == 3

    def test_set_tools_enabled_wildcard(self):
        pool = McpClientPool([])
        pool._tools["a.x"] = RegisteredTool(
            server_name="a", name="x", description="", input_schema={}
        )
        pool._tools["b.y"] = RegisteredTool(
            server_name="b", name="y", description="", input_schema={}
        )

        affected = pool.set_tools_enabled("*", False)
        assert len(affected) == 2
        assert len(pool.enabled_tool_schemas()) == 0

    def test_set_tools_enabled_bare_server_name(self):
        pool = McpClientPool([])
        pool._tools["acatome.paper"] = RegisteredTool(
            server_name="acatome",
            name="paper",
            description="Read",
            input_schema={},
        )
        pool._tools["acatome.search"] = RegisteredTool(
            server_name="acatome",
            name="search",
            description="Search",
            input_schema={},
        )
        pool._tools["precis.toc"] = RegisteredTool(
            server_name="precis",
            name="toc",
            description="TOC",
            input_schema={},
        )

        # Bare server name (no glob) should match all tools on that server
        affected = pool.set_tools_enabled("acatome", False)
        assert len(affected) == 2
        assert not pool._tools["acatome.paper"].enabled
        assert not pool._tools["acatome.search"].enabled
        assert pool._tools["precis.toc"].enabled

    def test_set_tools_enabled_bare_tool_name(self):
        pool = McpClientPool([])
        pool._tools["acatome.search"] = RegisteredTool(
            server_name="acatome",
            name="search",
            description="Search",
            input_schema={},
        )
        pool._tools["precis.toc"] = RegisteredTool(
            server_name="precis",
            name="toc",
            description="TOC",
            input_schema={},
        )

        # Bare tool name should match
        affected = pool.set_tools_enabled("search", False)
        assert len(affected) == 1
        assert not pool._tools["acatome.search"].enabled
        assert pool._tools["precis.toc"].enabled

    def test_tool_status(self):
        pool = McpClientPool([])
        pool._tools["s.t"] = RegisteredTool(
            server_name="s",
            name="t",
            description="A tool",
            input_schema={},
            enabled=False,
        )
        rows = pool.tool_status()
        assert len(rows) == 1
        assert rows[0]["name"] == "s.t"
        assert rows[0]["enabled"] is False


class TestCoerceArguments:
    """Test _coerce_arguments fixes double-encoded JSON from LLMs."""

    def test_array_as_string(self):
        schema = {
            "properties": {"messages": {"type": "array"}},
        }
        args = {"messages": '[{"role": "user", "content": "hello"}]'}
        fixed = McpClientPool._coerce_arguments(args, schema)
        assert isinstance(fixed["messages"], list)
        assert fixed["messages"][0]["role"] == "user"

    def test_object_as_string(self):
        schema = {
            "properties": {"config": {"type": "object"}},
        }
        args = {"config": '{"key": "value"}'}
        fixed = McpClientPool._coerce_arguments(args, schema)
        assert isinstance(fixed["config"], dict)
        assert fixed["config"]["key"] == "value"

    def test_already_correct(self):
        schema = {
            "properties": {"messages": {"type": "array"}},
        }
        args = {"messages": [{"role": "user", "content": "hello"}]}
        fixed = McpClientPool._coerce_arguments(args, schema)
        assert fixed["messages"] == args["messages"]

    def test_string_param_not_coerced(self):
        schema = {
            "properties": {"query": {"type": "string"}},
        }
        args = {"query": '["not", "an", "array"]'}
        fixed = McpClientPool._coerce_arguments(args, schema)
        assert isinstance(fixed["query"], str)

    def test_no_schema_passthrough(self):
        args = {"messages": '[{"role": "user"}]'}
        fixed = McpClientPool._coerce_arguments(args, {})
        assert isinstance(fixed["messages"], str)

    def test_invalid_json_ignored(self):
        schema = {
            "properties": {"data": {"type": "array"}},
        }
        args = {"data": "not json at all"}
        fixed = McpClientPool._coerce_arguments(args, schema)
        assert fixed["data"] == "not json at all"
