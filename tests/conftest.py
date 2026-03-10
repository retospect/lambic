"""Shared fixtures for lambic tests."""

from __future__ import annotations

import pytest

from lambic.core.config import LlmConfig, McpServer, ShellConfig


@pytest.fixture
def llm_config():
    return LlmConfig(provider="ollama", model="qwen3.5:9b", think=True)


@pytest.fixture
def shell_config(llm_config):
    return ShellConfig(
        llm=llm_config,
        servers=[],
        system_prompt="You are a test assistant.",
        max_tool_result=1000,
    )
