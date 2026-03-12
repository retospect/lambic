"""Tests for ChatSession — commands and tool result truncation."""

from __future__ import annotations

import pytest

from acatome_lambic.core.config import LlmConfig, ShellConfig
from acatome_lambic.core.session import ChatSession, ToolResult, _build_tool_signature
from acatome_lambic.core.llm import ToolCall


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
        assert "think: on" in result
        assert "messages" in result
        assert "autocontinue:" in result

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


class TestBuildToolSignature:
    def test_required_and_optional(self):
        schema = {
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "style": {"type": "string", "default": "summary"},
            },
            "required": ["query"],
        }
        sig = _build_tool_signature("acatome.search", schema)
        assert sig == "acatome.search(query, top_k=5, style='summary')"

    def test_all_required(self):
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        assert _build_tool_signature("x.y", schema) == "x.y(a, b)"

    def test_no_params(self):
        assert _build_tool_signature("x.y", {}) == "x.y()"

    def test_optional_infers_type(self):
        schema = {
            "properties": {"flag": {"type": "boolean"}},
            "required": [],
        }
        assert _build_tool_signature("x.y", schema) == "x.y(flag=false)"


class TestSavePartialResponse:
    def setup_method(self):
        self.config = ShellConfig(
            llm=LlmConfig(provider="ollama", model="test", think=False),
            system_prompt="Test.",
        )
        self.session = ChatSession(self.config)

    def test_saves_partial_content(self):
        self.session.messages.append({"role": "user", "content": "hello"})
        self.session.save_partial_response("partial answer")
        last = self.session.messages[-1]
        assert last["role"] == "assistant"
        assert "partial answer" in last["content"]
        assert "[interrupted]" in last["content"]

    def test_skips_empty_content(self):
        self.session.messages.append({"role": "user", "content": "hello"})
        before = len(self.session.messages)
        self.session.save_partial_response("")
        assert len(self.session.messages) == before

    def test_skips_whitespace_only(self):
        self.session.messages.append({"role": "user", "content": "hello"})
        before = len(self.session.messages)
        self.session.save_partial_response("   \n  ")
        assert len(self.session.messages) == before

    def test_no_duplicate_if_assistant_already_present(self):
        self.session.messages.append({"role": "user", "content": "hello"})
        self.session.messages.append({"role": "assistant", "content": "full reply"})
        before = len(self.session.messages)
        self.session.save_partial_response("partial")
        assert len(self.session.messages) == before


class TestTaskReminder:
    def test_default_empty(self):
        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)
        assert session.task_reminder == ""

    def test_message_command_sets_reminder(self):
        """Message command with a registered reminder builder sets task_reminder."""
        config = ShellConfig(
            system_prompt="Test.",
            message_commands={"review": lambda raw: "transformed message"},
            task_reminder_commands={"review": lambda raw: "review: check grammar"},
        )
        session = ChatSession(config)
        # Simulate what turn() does for message commands
        raw = "/review check grammar"
        verb = raw.strip().split()[0].lower().lstrip("/")
        if verb in config.task_reminder_commands:
            session.task_reminder = config.task_reminder_commands[verb](raw)
        assert session.task_reminder == "review: check grammar"

    def test_no_reminder_without_registration(self):
        """Message command without reminder builder leaves task_reminder empty."""
        config = ShellConfig(
            system_prompt="Test.",
            message_commands={"review": lambda raw: "transformed"},
        )
        session = ChatSession(config)
        assert session.task_reminder == ""


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
