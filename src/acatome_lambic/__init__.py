"""lambic — MCP-aware LLM shell with provider switching."""

from importlib.metadata import version

__version__ = version("acatome-lambic")

from acatome_lambic.core.config import LlmConfig, McpServer, ShellConfig
from acatome_lambic.core.session import ChatSession
from acatome_lambic.tui.app import Shell

__all__ = ["ChatSession", "LlmConfig", "McpServer", "Shell", "ShellConfig"]
