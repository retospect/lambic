# acatome-lambic

MCP-aware LLM shell with provider switching.

Connects to MCP servers via stdio, talks to LLMs (ollama, OpenAI, Anthropic via litellm),
and provides a terminal chat interface with tool calling.

## Usage

```python
from acatome_lambic.tui.app import Shell
from acatome_lambic.core.config import LlmConfig, McpServer, ShellConfig

config = ShellConfig(
    llm=LlmConfig(provider="ollama", model="qwen3.5:9b"),
    servers=[
        McpServer(name="precis", cmd=["uv", "run", "precis"]),
    ],
    system_prompt="You are a research assistant.",
)
Shell(config).run()
```

## Commands

- `/model <provider/model>` — switch LLM
- `/think on|off` — toggle reasoning mode (default: on)
- `/tools` — list tools with on/off status
- `/tools off <pattern>` — disable tools matching pattern
- `/tools on <pattern>` — enable tools matching pattern
- `/expand <call_id>` — show full (untruncated) tool result
- `/status` — show session info
- `/clear` — clear message history
- `/help` — show command help
- `/quit` — exit

Applications can register custom commands (`custom_commands`) and
LLM-routed message commands (`message_commands`) via `ShellConfig`.
All registered commands appear in tab autocomplete.
