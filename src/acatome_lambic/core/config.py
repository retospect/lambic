"""Configuration dataclasses for lambic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class LlmConfig:
    """LLM provider configuration."""

    provider: str = "ollama"
    model: str = "qwen3.5:9b"
    think: bool = True
    ollama_url: str = "http://localhost:11434"
    max_tokens: int = 16384
    temperature: float = 0.7
    api_keys: dict[str, str] = field(default_factory=dict)

    @property
    def spec(self) -> str:
        """litellm-style model spec, e.g. 'ollama/qwen3.5:9b'."""
        return f"{self.provider}/{self.model}"

    def ensure_api_keys(self) -> None:
        """Push configured API keys into env vars (if not already set)."""
        _KEY_MAP = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        for provider_key, env_var in _KEY_MAP.items():
            value = self.api_keys.get(provider_key, "")
            if value and not os.environ.get(env_var):
                os.environ[env_var] = value


@dataclass
class McpServer:
    """MCP server configuration."""

    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ShellConfig:
    """Top-level shell configuration."""

    llm: LlmConfig = field(default_factory=LlmConfig)
    servers: list[McpServer] = field(default_factory=list)
    system_prompt: str = ""
    max_tool_result: int = 8192
    log_file: str = ""
    message_commands: dict[str, Callable[[str], str]] = field(default_factory=dict)
    task_reminder_commands: dict[str, Callable[[str], str]] = field(
        default_factory=dict
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> ShellConfig:
        """Load config from a TOML file."""
        import tomllib

        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Extract top-level scalars first
        system_prompt = data.get("system_prompt", "")
        max_tool_result = data.get("max_tool_result", 8192)
        log_file = data.get("log_file", "")

        llm_data = data.get("llm", {})
        api_keys = llm_data.pop("api_keys", {})
        llm = LlmConfig(**llm_data, api_keys=api_keys)

        _server_keys = {f.name for f in McpServer.__dataclass_fields__.values()}
        servers = []
        for s in data.get("servers", []):
            filtered = {k: v for k, v in s.items() if k in _server_keys}
            servers.append(McpServer(**filtered))

        return cls(
            llm=llm,
            servers=servers,
            system_prompt=system_prompt,
            max_tool_result=max_tool_result,
            log_file=log_file,
        )

    @classmethod
    def default_path(cls) -> Path:
        """Default config file location."""
        return Path.home() / ".config" / "lambic" / "config.toml"
