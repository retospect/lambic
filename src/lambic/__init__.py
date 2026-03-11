"""lambic — MCP-aware LLM shell with provider switching."""

__version__ = "0.1.0"

from lambic.core.config import LlmConfig, McpServer, ShellConfig
from lambic.core.session import ChatSession
from lambic.tui.app import Shell

__all__ = ["LlmConfig", "McpServer", "ShellConfig", "ChatSession", "Shell"]
