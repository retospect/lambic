"""Tests for LLM shim."""

from __future__ import annotations

import pytest

from lambic.core.llm import (
    LlmClient,
    LlmResponse,
    ToolCall,
    tools_to_ollama,
)
from lambic.core.config import LlmConfig


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
