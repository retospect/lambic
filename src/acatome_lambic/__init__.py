"""lambic — MCP-aware LLM shell with provider switching."""

__version__ = "0.2.5"

from acatome_lambic.core.config import LlmConfig, McpServer, ShellConfig
from acatome_lambic.core.session import ChatSession
from acatome_lambic.tui.app import Shell

__all__ = ["LlmConfig", "McpServer", "ShellConfig", "ChatSession", "Shell"]
