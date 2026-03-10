"""Tests for ChatSession — commands and tool result truncation."""

from __future__ import annotations

import pytest

from lambic.core.config import LlmConfig, ShellConfig
from lambic.core.session import ChatSession, ToolResult
from lambic.core.llm import ToolCall


class TestCommands:
    def setup_method(self):
        self.config = ShellConfig(
            llm=LlmConfig(provider="ollama", model="qwen3.5:9b", think=True),
            system_prompt="Test assistant.",
            max_tool_result=100,
        )
        self.session = ChatSession(self.config)

    def test_system_prompt_in_messages(self):
        assert len(self.session.messages) == 1
        assert self.session.messages[0]["role"] == "system"
        assert self.session.messages[0]["content"] == "Test assistant."

    def test_no_system_prompt(self):
        cfg = ShellConfig(system_prompt="")
        s = ChatSession(cfg)
        assert len(s.messages) == 0

    def test_help(self):
        result = self.session._handle_command("/help")
        assert "/model" in result
        assert "/think" in result
        assert "/tools" in result
        assert "/quit" in result

    def test_model_show(self):
        result = self.session._handle_command("/model")
        assert "ollama/qwen3.5:9b" in result

    def test_model_switch(self):
        result = self.session._handle_command("/model openai/gpt-4o-mini")
        assert "openai/gpt-4o-mini" in result
        assert self.session.config.llm.provider == "openai"
        assert self.session.config.llm.model == "gpt-4o-mini"

    def test_model_switch_no_provider(self):
        result = self.session._handle_command("/model llama3.2:3b")
        assert "ollama/llama3.2:3b" in result

    def test_think_show(self):
        result = self.session._handle_command("/think")
        assert "on" in result

    def test_think_off(self):
        result = self.session._handle_command("/think off")
        assert "off" in result
        assert self.session.config.llm.think is False

    def test_think_on(self):
        self.session.config.llm.think = False
        result = self.session._handle_command("/think on")
        assert "on" in result
        assert self.session.config.llm.think is True

    def test_status(self):
        result = self.session._handle_command("/status")
        assert "ollama/qwen3.5:9b" in result
        assert "Think:" in result
        assert "Messages:" in result

    def test_clear(self):
        self.session.messages.append({"role": "user", "content": "hi"})
        self.session.messages.append({"role": "assistant", "content": "hello"})
        assert len(self.session.messages) == 3
        result = self.session._handle_command("/clear")
        assert "cleared" in result.lower()
        # System prompt preserved
        assert len(self.session.messages) == 1
        assert self.session.messages[0]["role"] == "system"

    def test_quit(self):
        result = self.session._handle_command("/quit")
        assert result == "__QUIT__"

    def test_exit(self):
        result = self.session._handle_command("/exit")
        assert result == "__QUIT__"

    def test_unknown_command(self):
        result = self.session._handle_command("/foobar")
        assert "Unknown command" in result

    def test_tools_empty(self):
        result = self.session._handle_command("/tools")
        assert "No tools" in result

    def test_expand_missing(self):
        result = self.session._handle_command("/expand tc_999")
        assert "No stored result" in result


class TestTruncation:
    def test_tool_result_truncation(self):
        """Verify max_tool_result is respected."""
        config = ShellConfig(max_tool_result=50)
        session = ChatSession(config)

        # Simulate a long result
        long_text = "x" * 200
        truncated = long_text[: config.max_tool_result]
        assert len(truncated) == 50
        assert len(long_text) > config.max_tool_result
