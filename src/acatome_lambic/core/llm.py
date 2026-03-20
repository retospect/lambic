"""LLM provider shim — ollama direct (httpx) + litellm fallback.

Supports streaming text and tool calling for both backends.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from acatome_lambic.core.config import LlmConfig

log = logging.getLogger("lambic.llm")


def _merge_concatenated_json(raw: str) -> dict[str, Any] | None:
    """Try to parse concatenated JSON objects and merge into one dict.

    Models like qwen concatenate multiple JSON objects when they want to
    batch calls: ``{"id":"a#1"}{"id":"b#2"}``.  This splits them apart
    and merges: string values with the same key are joined with commas,
    giving ``{"id": "a#1,b#2"}``.  Returns None if splitting fails.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    pos = 0
    s = raw.strip()
    while pos < len(s):
        # skip whitespace
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        if pos >= len(s):
            break
        try:
            obj, end = decoder.raw_decode(s, pos)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        objects.append(obj)
        pos = end
    if len(objects) < 2:
        return None  # not concatenated — let caller handle

    # Merge: join string values with commas, first-wins for others
    merged: dict[str, Any] = {}
    for obj in objects:
        for k, v in obj.items():
            if k not in merged:
                merged[k] = v
            elif isinstance(merged[k], str) and isinstance(v, str):
                merged[k] += "," + v
            # else: keep first value
    return merged


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmResponse:
    """Full (non-streaming) LLM response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = ""  # "stop", "length", "tool_calls", etc.


def tools_to_ollama(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool schemas to ollama tool format."""
    result = []
    for t in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object"}),
                },
            }
        )
    return result


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    """Make tool name safe for Anthropic: [a-zA-Z0-9_-]{1,128}."""
    return _SAFE_NAME_RE.sub("-", name)[:128]


def tools_to_litellm(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert MCP tool schemas to litellm/OpenAI tool format.

    Returns (tools_list, reverse_name_map) where reverse_name_map
    maps sanitized names back to original MCP names.
    """
    result = []
    reverse_map: dict[str, str] = {}  # sanitized -> original
    for t in tools:
        original = t["name"]
        safe = _sanitize_tool_name(original)
        reverse_map[safe] = original
        result.append(
            {
                "type": "function",
                "function": {
                    "name": safe,
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object"}),
                },
            }
        )
    return result, reverse_map


class LlmClient:
    """Unified LLM client with provider switching."""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        self._http: httpx.AsyncClient | None = None
        self._name_map: dict[str, str] = {}  # sanitized -> original tool name

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    @property
    def is_ollama(self) -> bool:
        return self.config.provider == "ollama"

    async def check_connection(self) -> bool:
        """Check if the LLM backend is reachable and model is available."""
        if self.is_ollama:
            try:
                http = await self._get_http()
                resp = await http.get(f"{self.config.ollama_url}/api/tags")
                resp.raise_for_status()
                models = {
                    m["name"] for m in resp.json().get("models", [])
                }
                target = self.config.model
                if target not in models and f"{target}:latest" not in models:
                    log.warning(
                        "Model %r not found locally, pulling from ollama...",
                        target,
                    )
                    try:
                        async with http.stream(
                            "POST",
                            f"{self.config.ollama_url}/api/pull",
                            json={"name": target},
                            timeout=httpx.Timeout(600.0),
                        ) as pull_resp:
                            if pull_resp.status_code != 200:
                                log.error("Failed to pull model %r", target)
                                return False
                            async for line in pull_resp.aiter_lines():
                                try:
                                    msg = json.loads(line)
                                    status = msg.get("status", "")
                                    if "completed" in msg and "total" in msg:
                                        pct = int(msg["completed"] / msg["total"] * 100)
                                        log.info("Pulling %s: %s %d%%", target, status, pct)
                                    elif status:
                                        log.info("Pulling %s: %s", target, status)
                                except (json.JSONDecodeError, KeyError, ZeroDivisionError):
                                    pass
                    except Exception as exc:
                        log.error("Failed to pull model %r: %s", target, exc)
                        return False
                return True
            except Exception:
                return False
        return True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LlmResponse:
        """Non-streaming completion with optional tool calling."""
        return await self._litellm_complete(messages, tools)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str | LlmResponse]:
        """Streaming completion.

        Yields str chunks for text content.
        If the LLM returns tool calls, yields a final LlmResponse with tool_calls.
        """
        async for item in self._litellm_stream(messages, tools):
            yield item

    # ── Ollama backend ──────────────────────────────────────────────

    @staticmethod
    def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Ollama's expected format.

        Ollama differences:
        - tool_calls: no 'id'/'type', arguments is a dict (not JSON string)
        - tool results: no 'tool_call_id'
        """
        result = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg["role"], "content": msg.get("content", "")}

            if "tool_calls" in msg:
                converted = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    converted.append(
                        {"function": {"name": fn["name"], "arguments": args}}
                    )
                m["tool_calls"] = converted

            # Ollama ignores tool_call_id — just pass role + content
            result.append(m)
        return result

    def _ollama_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._to_ollama_messages(messages),
            "stream": stream,
            "think": self.config.think,
        }
        if tools:
            payload["tools"] = tools_to_ollama(tools)
        payload["options"] = {"num_predict": self.config.max_tokens}
        return payload

    async def _ollama_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LlmResponse:
        http = await self._get_http()
        payload = self._ollama_payload(messages, tools, stream=False)
        resp = await http.post(f"{self.config.ollama_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_ollama_message(data.get("message", {}))

    async def _ollama_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncIterator[str | LlmResponse]:
        http = await self._get_http()
        payload = self._ollama_payload(messages, tools, stream=True)

        accumulated_tool_calls: list[dict] = []
        accumulated_content = ""
        thinking = ""

        async with http.stream(
            "POST", f"{self.config.ollama_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})

                # Thinking content (qwen3.5 reasoning)
                if msg.get("thinking"):
                    thinking += msg["thinking"]

                # Text content
                content = msg.get("content", "")
                if content:
                    accumulated_content += content
                    yield content

                # Tool calls accumulate
                if msg.get("tool_calls"):
                    accumulated_tool_calls.extend(msg["tool_calls"])

                # Final chunk
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                    stop_reason = (
                        "tool_calls"
                        if accumulated_tool_calls
                        else chunk.get("done_reason", "stop")
                    )
                    if accumulated_tool_calls:
                        yield LlmResponse(
                            content=accumulated_content,
                            tool_calls=self._parse_ollama_tool_calls(
                                accumulated_tool_calls
                            ),
                            thinking=thinking,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            stop_reason=stop_reason,
                        )
                    else:
                        yield LlmResponse(
                            content=accumulated_content,
                            thinking=thinking,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            stop_reason=stop_reason,
                        )
                    break

    def _parse_ollama_message(self, msg: dict) -> LlmResponse:
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        tool_calls = self._parse_ollama_tool_calls(msg.get("tool_calls", []))
        return LlmResponse(content=content, tool_calls=tool_calls, thinking=thinking)

    def _parse_ollama_tool_calls(self, raw: list[dict]) -> list[ToolCall]:
        calls = []
        for i, tc in enumerate(raw):
            fn = tc.get("function", {})
            calls.append(
                ToolCall(
                    id=f"call_{i}",
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", {}),
                )
            )
        return calls

    # ── litellm backend ─────────────────────────────────────────────

    async def _litellm_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LlmResponse:
        import litellm

        self.config.ensure_api_keys()
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        think = self.config.think if self.is_ollama else False
        log.info("%s  think=%s  tools=%d", self.config.spec, think, len(tools or []))

        kwargs: dict[str, Any] = {
            "model": self.config.spec,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.is_ollama:
            kwargs["api_base"] = self.config.ollama_url
            kwargs["think"] = self.config.think
        if tools:
            litellm_tools, self._name_map = tools_to_litellm(tools)
            kwargs["tools"] = litellm_tools

        resp = await litellm.acompletion(num_retries=3, **kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                name = self._name_map.get(tc.function.name, tc.function.name)
                tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{len(tool_calls)}",
                        name=name,
                        arguments=args,
                    )
                )

        return LlmResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            thinking=getattr(msg, "reasoning_content", "") or "",
        )

    async def _litellm_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AsyncIterator[str | LlmResponse]:
        import litellm

        self.config.ensure_api_keys()
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        think = self.config.think if self.is_ollama else False
        log.info("%s  think=%s  tools=%d  stream", self.config.spec, think, len(tools or []))

        kwargs: dict[str, Any] = {
            "model": self.config.spec,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if self.is_ollama:
            kwargs["api_base"] = self.config.ollama_url
            kwargs["think"] = self.config.think
        if tools:
            litellm_tools, self._name_map = tools_to_litellm(tools)
            kwargs["tools"] = litellm_tools

        resp = await litellm.acompletion(num_retries=3, **kwargs)

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in resp:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta

            # Capture finish_reason from the final chunk
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            if delta and delta.content:
                accumulated_content += delta.content
                yield delta.content

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id or f"call_{idx}",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx][
                                "arguments"
                            ] += tc_delta.function.arguments

            # Extract usage from the final chunk (litellm puts it on the chunk)
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Normalize stop reason
        stop_reason = finish_reason or (
            "tool_calls" if accumulated_tool_calls else "stop"
        )
        if stop_reason == "end_turn":
            stop_reason = "stop"

        if accumulated_tool_calls:
            tool_calls = []
            for idx in sorted(accumulated_tool_calls):
                tc = accumulated_tool_calls[idx]
                args = tc["arguments"]
                log.info(
                    "RAW tool_call[%d]: name=%r args_type=%s args=%r",
                    idx, tc["name"], type(args).__name__, args[:500] if isinstance(args, str) else args,
                )
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        merged = _merge_concatenated_json(args)
                        if merged is not None:
                            log.info("Merged %d concatenated JSON objects", len(args.split("}{")) )
                            args = merged
                        else:
                            log.warning("Failed to parse tool args JSON: %r", args[:200])
                            args = {"__parse_error__": args}
                log.info("PARSED tool_call[%d]: name=%r args=%r", idx, tc["name"], args)
                name = self._name_map.get(tc["name"], tc["name"])
                tool_calls.append(ToolCall(id=tc["id"], name=name, arguments=args))
            yield LlmResponse(
                content=accumulated_content,
                tool_calls=tool_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                stop_reason=stop_reason,
            )
        else:
            yield LlmResponse(
                content=accumulated_content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                stop_reason=stop_reason,
            )
