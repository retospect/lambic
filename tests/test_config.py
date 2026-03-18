"""Tests for lambic configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from acatome_lambic.core.config import LlmConfig, McpServer, ShellConfig


class TestLlmConfig:
    def test_defaults(self):
        cfg = LlmConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen3.5:9b"
        assert cfg.think is True
        assert cfg.spec == "ollama_chat/qwen3.5:9b"

    def test_spec(self):
        cfg = LlmConfig(provider="openai", model="gpt-4o-mini")
        assert cfg.spec == "openai/gpt-4o-mini"

    def test_ensure_api_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LlmConfig(api_keys={"openai": "sk-test123"})
        cfg.ensure_api_keys()
        assert os.environ["OPENAI_API_KEY"] == "sk-test123"

    def test_ensure_api_keys_no_overwrite(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "existing")
        cfg = LlmConfig(api_keys={"openai": "sk-new"})
        cfg.ensure_api_keys()
        assert os.environ["OPENAI_API_KEY"] == "existing"


class TestMcpServer:
    def test_create(self):
        s = McpServer(name="test", cmd=["uv", "run", "test-mcp"])
        assert s.name == "test"
        assert s.enabled is True
        assert s.env == {}


class TestShellConfig:
    def test_defaults(self):
        cfg = ShellConfig()
        assert cfg.llm.provider == "ollama"
        assert cfg.servers == []
        assert cfg.max_tool_result == 8192

    def test_from_toml(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("""
system_prompt = "Hello world"
max_tool_result = 4096

[llm]
provider = "openai"
model = "gpt-4o-mini"
think = false

[llm.api_keys]
openai = "sk-test"

[[servers]]
name = "test"
cmd = ["echo", "hello"]
""")
        cfg = ShellConfig.from_toml(toml_file)
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.llm.think is False
        assert cfg.llm.api_keys == {"openai": "sk-test"}
        assert len(cfg.servers) == 1
        assert cfg.servers[0].name == "test"
        assert cfg.system_prompt == "Hello world"
        assert cfg.max_tool_result == 4096

    def test_from_toml_missing_file(self, tmp_path):
        cfg = ShellConfig.from_toml(tmp_path / "nonexistent.toml")
        assert cfg.llm.provider == "ollama"

    def test_default_path(self):
        p = ShellConfig.default_path()
        assert p.name == "config.toml"
        assert "lambic" in str(p)
