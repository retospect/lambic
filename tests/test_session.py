"""Tests for ChatSession — commands and tool result truncation."""

from __future__ import annotations

from acatome_lambic.core.config import LlmConfig, ShellConfig
from acatome_lambic.core.llm import ToolCall
from acatome_lambic.core.session import ChatSession, _build_tool_signature


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
        assert "ollama_chat/qwen3.5:9b" in result

    def test_model_switch(self):
        result = self.session._handle_command("/model openai/gpt-4o-mini")
        assert "openai/gpt-4o-mini" in result
        assert self.session.config.llm.provider == "openai"
        assert self.session.config.llm.model == "gpt-4o-mini"

    def test_model_switch_no_provider(self):
        result = self.session._handle_command("/model llama3.2:3b")
        assert "ollama_chat/llama3.2:3b" in result

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
        assert "ollama_chat/qwen3.5:9b" in result
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


class TestCustomCommands:
    def test_custom_command_dispatched(self):
        config = ShellConfig(
            system_prompt="Test.",
            custom_commands={"db": lambda cmd: "db info here"},
        )
        session = ChatSession(config)
        result = session._handle_command("/db")
        assert result == "db info here"

    def test_custom_command_shown_in_help(self):
        config = ShellConfig(
            system_prompt="Test.",
            custom_commands={"db": lambda cmd: ""},
        )
        session = ChatSession(config)
        result = session._handle_command("/help")
        assert "/db" in result

    def test_unknown_without_custom(self):
        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)
        result = session._handle_command("/db")
        assert "Unknown command" in result


class TestParseErrorInterception:
    """Tests for malformed JSON tool call handling in session."""

    def test_parse_error_returns_error_result(self):
        """__parse_error__ in args should produce ERROR result without calling tool."""
        import asyncio

        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)

        tc = ToolCall(
            id="call_1",
            name="precis.get",
            arguments={"__parse_error__": '{"id":"a"}{"id":"b"}'},
        )
        results = asyncio.run(session._execute_tools([tc]))
        assert len(results) == 1
        assert "ERROR" in results[0].result
        assert "malformed JSON" in results[0].result
        assert "comma-separated" in results[0].result

    def test_extract_last_hint_finds_get(self):
        """Should extract get(id='...') from recent successful tool results."""
        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)
        session.messages.append({"role": "user", "content": "search MOF"})
        session.messages.append(
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": (
                    "5 results for: MOF\n"
                    "  chen2020~23  (0.16)\n\n"
                    "Next:\n"
                    "  get(id='chen2020~23')  — read this chunk\n"
                    "  get(id='chen2020/toc')  — paper structure"
                ),
            }
        )
        hint = session._extract_last_hint()
        assert hint == "get(id='chen2020~23')"

    def test_extract_last_hint_skips_errors(self):
        """Should skip ERROR tool results and find hints from earlier results."""
        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)
        session.messages.append({"role": "user", "content": "search MOF"})
        session.messages.append(
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "Next:\n  get(id='slug1~5')  — read",
            }
        )
        session.messages.append(
            {
                "role": "tool",
                "tool_call_id": "tc_2",
                "content": "ERROR: id required",
            }
        )
        hint = session._extract_last_hint()
        assert hint == "get(id='slug1~5')"

    def test_extract_last_hint_empty_when_no_hints(self):
        """Should return empty string when no hints found."""
        config = ShellConfig(system_prompt="Test.")
        session = ChatSession(config)
        session.messages.append({"role": "user", "content": "hello"})
        assert session._extract_last_hint() == ""


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
