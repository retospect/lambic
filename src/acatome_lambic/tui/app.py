"""Terminal UI — prompt_toolkit input + rich output."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import weakref
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from acatome_lambic.core.config import ShellConfig
from acatome_lambic.core.session import ChatSession

log = logging.getLogger("lambic.tui")

# Slash commands for tab completion
_SLASH_COMMANDS = [
    ("/model", "switch LLM (provider/model)"),
    ("/think", "toggle reasoning mode (on|off)"),
    ("/tools", "list/toggle tools (on|off pattern)"),
    ("/max_tokens", "show/set output token limit"),
    ("/more", "continue generation (opt. N tokens)"),
    ("/autocontinue", "autocontinue on truncation (on|off)"),
    ("/expand", "show full tool result (call_id)"),
    ("/status", "session info"),
    ("/prompt", "dump full LLM prompt"),
    ("/clear", "clear history"),
    ("/quit", "exit"),
    ("/help", "show commands"),
]

# Subcommand options for tab completion (command → list of options)
_SLASH_SUBOPTIONS: dict[str, list[str]] = {
    "/think": ["on", "off"],
    "/tools": ["on", "off"],
    "/autocontinue": ["on", "off"],
}


class _SlashCompleter(Completer):
    """Tab-complete /commands and their subcommand options."""

    def __init__(self, session_ref: callable = None):
        self._session_ref = session_ref

    def _tool_names(self) -> list[str]:
        """Get registered tool names from the live session."""
        if self._session_ref is None:
            return []
        session = self._session_ref()
        if session is None:
            return []
        return list(session.mcp._tools.keys())

    def _model_names(self) -> list[str]:
        """Get available model names for /model completion."""
        models: list[str] = []
        # Ollama models (cached, fetched once)
        if not hasattr(self, "_ollama_cache"):
            self._ollama_cache: list[str] = []
            try:
                import httpx

                resp = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
                if resp.status_code == 200:
                    for m in resp.json().get("models", []):
                        self._ollama_cache.append(m["name"])
            except Exception:
                pass
        models.extend(self._ollama_cache)
        # Claude models if API key present
        if os.environ.get("ANTHROPIC_API_KEY"):
            models.extend(
                [
                    "anthropic/claude-sonnet-4-20250514",
                    "anthropic/claude-3-5-haiku-20241022",
                ]
            )
        # OpenAI models if API key present
        if os.environ.get("OPENAI_API_KEY"):
            models.extend(
                [
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                ]
            )
        return models

    def _extra_commands(self) -> list[tuple[str, str]]:
        """Get custom and message commands from the live session config."""
        if self._session_ref is None:
            return []
        session = self._session_ref()
        if session is None:
            return []
        extras = []
        for name in session.config.custom_commands:
            extras.append((f"/{name}", "custom command"))
        for name in session.config.message_commands:
            extras.append((f"/{name}", "→ LLM"))
        return extras

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        if " " not in text:
            # First word — complete command name
            all_commands = _SLASH_COMMANDS + self._extra_commands()
            for cmd, desc in all_commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display_meta=desc,
                    )
        else:
            # Complete the last word being typed
            cmd = text.split()[0]
            words = text.split()
            word = words[-1] if len(words) > 1 and not text.endswith(" ") else ""
            # First argument: subcommand options + tool names
            # Subsequent arguments: tool/server names (for multi-pattern)
            options = []
            if len(words) <= 2 and not text.endswith(" "):
                options = list(_SLASH_SUBOPTIONS.get(cmd, []))
            if cmd == "/tools":
                options.extend(self._tool_names())
                # Also offer server names for bulk toggling
                options.extend(sorted({n.split(".")[0] for n in self._tool_names()}))
            if cmd == "/model":
                options.extend(self._model_names())
            for opt in sorted(set(options)):
                if opt.startswith(word):
                    yield Completion(opt, start_position=-len(word))


def _make_key_bindings() -> KeyBindings:
    """Enter submits, Alt+Enter (Esc Enter) inserts newline."""
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return kb


class Shell:
    """Terminal chat shell."""

    def __init__(
        self,
        config: ShellConfig | None = None,
        *,
        model: Any = None,
        servers: list[Any] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        if config is None:
            from acatome_lambic.core.config import LlmConfig

            llm = model if isinstance(model, LlmConfig) else LlmConfig()
            svrs = servers or []
            config = ShellConfig(
                llm=llm,
                servers=svrs,
                system_prompt=system_prompt,
                **kwargs,
            )
        self.config = config
        self.console = Console()
        self.session: ChatSession | None = None

    def run(self) -> None:
        """Synchronous entry point."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self.console.print("\n[dim]Goodbye.[/dim]")

    async def _run_async(self) -> None:
        self.session = ChatSession(self.config)
        self._turn_task: asyncio.Task | None = None

        # Connect
        self.console.print(
            Panel(
                "[bold]lambic[/bold] — MCP-aware LLM shell",
                style="blue",
                expand=False,
            )
        )
        self.console.print(
            f"[dim]Connecting to [bold]{self.config.llm.spec}[/bold]...[/dim]"
        )

        status = await self.session.start()
        self._print_status(status)

        # Prompt loop — pass weak ref to session for dynamic tool completion
        session_ref = weakref.ref(self.session)
        prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            completer=_SlashCompleter(session_ref),
            key_bindings=_make_key_bindings(),
            multiline=True,
        )

        # Install SIGINT handler that cancels the current turn task
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        except NotImplementedError:
            pass  # Windows: signal handlers not supported, KeyboardInterrupt still works

        try:
            while True:
                try:
                    user_input = await loop.run_in_executor(
                        None,
                        lambda: prompt_session.prompt(
                            HTML("<b><ansigreen>› </ansigreen></b>")
                        ),
                    )
                except EOFError:
                    break
                except KeyboardInterrupt:
                    self.console.print("\n[dim]  /quit to exit[/dim]")
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                result = await self._run_turn(user_input)
                if result == "__QUIT__":
                    break
                if result == "__MORE__":
                    await self._run_turn("Continue from where you left off.")
                    # Restore max_tokens if it was temporarily changed
                    session = self.session
                    if session and hasattr(session, "_more_restore_tokens"):
                        if session._more_extra:
                            session.config.llm.max_tokens = session._more_restore_tokens
                            session.llm.config.max_tokens = session._more_restore_tokens

        finally:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except NotImplementedError:
                pass  # Windows
            await self.session.close()

    async def _run_turn(self, user_input: str) -> str | None:
        """Run _process_turn as a cancellable child task."""
        self._turn_task = asyncio.create_task(self._process_turn(user_input))
        try:
            return await self._turn_task
        except asyncio.CancelledError:
            # CancelledError escaped _process_turn (rare edge case)
            if self.session:
                self.session.save_partial_response("")
            self.console.print("\n[dim italic]  ⏎ interrupted[/dim italic]\n")
            return None
        finally:
            self._turn_task = None

    def _on_sigint(self) -> None:
        """Handle SIGINT (Ctrl-C) inside the event loop."""
        if self._turn_task is not None:
            self._turn_task.cancel()
        else:
            # At the prompt — print hint (prompt_toolkit handles the rest)
            self.console.print("\n[dim]  /quit to exit[/dim]")

    def _stop_live(self, live: Live | None, md_buffer: str) -> None:
        """Finalize and stop a Live markdown render."""
        if live is None:
            return
        if md_buffer.strip():
            live.update(Markdown(md_buffer))
        live.stop()

    async def _process_turn(self, user_input: str) -> str | None:
        """Process one user turn, rendering events to the terminal.

        Returns '__MORE__' if /more was invoked, None otherwise.
        Ctrl-C during streaming interrupts cleanly: partial response is
        saved to history and the prompt returns.
        """
        assert self.session is not None

        md_buffer = ""
        live: Live | None = None
        more_signal = None

        try:
            async for event in self.session.turn(user_input):
                if event.kind == "text":
                    if event.data == "__QUIT__":
                        self._stop_live(live, md_buffer)
                        self.console.print("[dim]Goodbye.[/dim]")
                        raise SystemExit(0)

                    if event.data == "__MORE__":
                        more_signal = "__MORE__"
                        continue

                    md_buffer += event.data
                    if live is None:
                        live = Live(
                            Markdown(md_buffer),
                            console=self.console,
                            refresh_per_second=4,
                        )
                        live.start()
                    else:
                        live.update(Markdown(md_buffer))

                elif event.kind == "tool_result":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self._print_tool_result(event.data)

                elif event.kind == "thinking":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self.console.print(
                        Panel(
                            Text(event.data[:500], style="dim italic"),
                            title="[dim]thinking[/dim]",
                            border_style="dim",
                            expand=False,
                        )
                    )

                elif event.kind == "command":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    if event.data == "__QUIT__":
                        self.console.print("[dim]Goodbye.[/dim]")
                        return "__QUIT__"
                    if event.data == "__MORE__":
                        more_signal = "__MORE__"
                        continue
                    self.console.print(event.data)

                elif event.kind == "status":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self.console.print(f"[dim yellow]{event.data}[/dim yellow]")

                elif event.kind == "error":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self.console.print(f"[bold red]Error:[/bold red] {event.data}")

                elif event.kind == "usage":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self._print_usage(event.data)

                elif event.kind == "done":
                    self._stop_live(live, md_buffer)
                    live = None
                    md_buffer = ""
                    self.console.print()  # blank line after response

        except (KeyboardInterrupt, asyncio.CancelledError):
            self._stop_live(live, md_buffer)
            self.session.save_partial_response(md_buffer)
            self.console.print("\n[dim italic]  ⏎ interrupted[/dim italic]\n")

        return more_signal

    def _print_usage(self, usage: dict) -> None:
        """Show token usage and truncation hint."""
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        stop = usage.get("stop_reason", "")
        max_tok = usage.get("max_tokens", 0)
        auto = usage.get("autocontinue", False)

        parts = []
        if prompt or completion:
            parts.append(f"{prompt}→{completion} tokens")
        if stop == "length":
            if auto:
                parts.append(
                    f"[bold yellow]⚠ hit max_tokens ({max_tok})[/bold yellow] "
                    f"— auto-continuing…"
                )
            else:
                parts.append(
                    f"[bold yellow]⚠ hit max_tokens ({max_tok})[/bold yellow] "
                    f"— type [bold]/more[/bold] to continue"
                )
        elif auto:
            parts.append("[dim cyan]↻ auto-continuing…[/dim cyan]")

        if parts:
            self.console.print(f"[dim]  {' · '.join(parts)}[/dim]")

    def _print_tool_result(self, tr: Any) -> None:
        """Render a tool call + result."""
        # Show the call
        args_str = ", ".join(f"{k}={v!r}" for k, v in tr.call.arguments.items())
        call_text = f"{tr.call.name}({args_str})"

        # Truncation indicator
        trunc = " [truncated]" if tr.truncated else ""
        timing = f"{tr.elapsed:.1f}s"

        self.console.print(
            Panel(
                Text(call_text, style="bold cyan"),
                title=f"[dim]tool call · {timing}{trunc}[/dim]",
                border_style="cyan",
                expand=False,
            )
        )

        self.console.print(
            Panel(
                tr.result,
                title="[dim]result[/dim]",
                border_style="dim",
                expand=False,
            )
        )

    def _print_status(self, status: dict) -> None:
        """Print connection status."""
        llm_icon = "●" if status["llm_ok"] else "○"
        llm_color = "green" if status["llm_ok"] else "red"
        think = "on" if status["think"] else "off"

        self.console.print(
            f"  [{llm_color}]{llm_icon}[/{llm_color}] LLM: "
            f"[bold]{status['model']}[/bold] (think: {think})"
        )

        for name, srv_status in status.get("servers", {}).items():
            icon = "●" if srv_status == "connected" else "○"
            color = "green" if srv_status == "connected" else "red"
            self.console.print(f"  [{color}]{icon}[/{color}] {name}: {srv_status}")

        self.console.print(
            f"  [dim]{status['tools']} tools available. "
            f"Type /help for commands.[/dim]\n"
        )
