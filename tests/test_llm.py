"""Tests for LLM shim."""

from __future__ import annotations

import pytest

from acatome_lambic.core.config import LlmConfig
from acatome_lambic.core.llm import (
    LlmClient,
    LlmResponse,
    ToolCall,
    tools_to_ollama,
)


class TestToolConversion:
    def test_tools_to_ollama(self):
        mcp_tools = [
            {
                "name": "paper",
                "description": "Read a paper",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                    },
                    "required": ["id"],
                },
            }
        ]
        result = tools_to_ollama(mcp_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "paper"
        assert result[0]["function"]["description"] == "Read a paper"
        assert "properties" in result[0]["function"]["parameters"]

    def test_tools_to_ollama_empty(self):
        assert tools_to_ollama([]) == []

    def test_tools_to_ollama_missing_schema(self):
        result = tools_to_ollama([{"name": "test"}])
        assert result[0]["function"]["parameters"] == {"type": "object"}


class TestLlmResponse:
    def test_text_only(self):
        r = LlmResponse(content="Hello world")
        assert r.content == "Hello world"
        assert r.tool_calls == []

    def test_with_tool_calls(self):
        r = LlmResponse(
            content="",
            tool_calls=[
                ToolCall(id="1", name="paper", arguments={"id": "slug:x"}),
            ],
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "paper"


class TestLlmClient:
    def test_is_ollama(self):
        cfg = LlmConfig(provider="ollama")
        client = LlmClient(cfg)
        assert client.is_ollama is True

    def test_is_not_ollama(self):
        cfg = LlmConfig(provider="openai")
        client = LlmClient(cfg)
        assert client.is_ollama is False

    def test_ollama_payload(self):
        cfg = LlmConfig(provider="ollama", model="qwen3.5:9b", think=True)
        client = LlmClient(cfg)
        payload = client._ollama_payload(
            [{"role": "user", "content": "hi"}], None, stream=False
        )
        assert payload["model"] == "qwen3.5:9b"
        assert payload["think"] is True
        assert payload["stream"] is False
        assert "tools" not in payload

    def test_ollama_payload_with_tools(self):
        cfg = LlmConfig(provider="ollama", model="qwen3.5:9b")
        client = LlmClient(cfg)
        tools = [
            {"name": "test", "description": "t", "inputSchema": {"type": "object"}}
        ]
        payload = client._ollama_payload(
            [{"role": "user", "content": "hi"}], tools, stream=True
        )
        assert "tools" in payload
        assert payload["stream"] is True

    def test_parse_ollama_tool_calls(self):
        cfg = LlmConfig()
        client = LlmClient(cfg)
        raw = [
            {"function": {"name": "paper", "arguments": {"id": "slug:x"}}},
            {"function": {"name": "search", "arguments": {"query": "MOF"}}},
        ]
        calls = client._parse_ollama_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].name == "paper"
        assert calls[1].arguments == {"query": "MOF"}

    def test_parse_ollama_message(self):
        cfg = LlmConfig()
        client = LlmClient(cfg)
        msg = {
            "content": "Here is the result",
            "thinking": "Let me think...",
            "tool_calls": [],
        }
        resp = client._parse_ollama_message(msg)
        assert resp.content == "Here is the result"
        assert resp.thinking == "Let me think..."
        assert resp.tool_calls == []


class TestConcatenatedJsonParsing:
    """Tests for malformed tool call arguments from the LLM."""

    def test_valid_json_parsed(self):
        """Normal single JSON object should parse fine."""
        import json

        args_str = '{"id": "wang2020state~5"}'
        args = json.loads(args_str)
        assert args == {"id": "wang2020state~5"}

    def test_concatenated_json_fails_loads(self):
        """Multiple JSON objects concatenated should fail json.loads."""
        import json

        args_str = '{"id": "slug1~4"}{"id": "slug2~9"}{"query": "MOF"}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(args_str)

    def test_merge_two_ids(self):
        """Two concatenated id objects should merge with comma."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        result = _merge_concatenated_json('{"id": "slug1~4"}{"id": "slug2~9"}')
        assert result == {"id": "slug1~4,slug2~9"}

    def test_merge_many_ids(self):
        """Many concatenated id objects should all merge."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        raw = '{"id": "a~1"}{"id": "b~2"}{"id": "c~3"}{"id": "d~4"}'
        result = _merge_concatenated_json(raw)
        assert result == {"id": "a~1,b~2,c~3,d~4"}

    def test_merge_mixed_keys(self):
        """Mixed keys: id merges, query takes first."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        raw = '{"id": "slug1~4"}{"id": "slug2~9"}{"query": "MOF"}'
        result = _merge_concatenated_json(raw)
        assert result["id"] == "slug1~4,slug2~9"
        assert result["query"] == "MOF"

    def test_merge_single_object_returns_none(self):
        """Single valid JSON should return None (not concatenated)."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        assert _merge_concatenated_json('{"id": "slug1~4"}') is None

    def test_merge_garbage_returns_none(self):
        """Non-JSON garbage should return None."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        assert _merge_concatenated_json("not json at all") is None

    def test_merge_empty_returns_none(self):
        """Empty string should return None."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        assert _merge_concatenated_json("") is None

    def test_merge_with_range(self):
        """Real-world case: range + single IDs."""
        from acatome_lambic.core.llm import _merge_concatenated_json

        raw = '{"id": "piscopo2020strategies~60..70"}{"id": "liu2019trace~5"}{"id": "kim2017beyond~12"}'
        result = _merge_concatenated_json(raw)
        assert result == {
            "id": "piscopo2020strategies~60..70,liu2019trace~5,kim2017beyond~12"
        }

    def test_empty_string_produces_parse_error(self):
        """Empty string args should still flag parse error (not concatenated)."""
        import json

        from acatome_lambic.core.llm import _merge_concatenated_json

        args_str = ""
        merged = _merge_concatenated_json(args_str)
        assert merged is None  # not concatenated
        with pytest.raises(json.JSONDecodeError):
            json.loads(args_str)
