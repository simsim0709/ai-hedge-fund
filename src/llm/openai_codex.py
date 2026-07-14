"""Experimental ChatGPT OAuth adapter for the Codex Responses endpoint.

This provider is deliberately separate from the standard OpenAI Platform API
key path.  It uses an OAuth token managed by ``oauth-cli-kit`` and only sends
that token to the allowlisted ChatGPT Codex endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, BaseMessage

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_CODEX_MODEL = "openai-codex/gpt-5.4"
LOGIN_COMMAND = "poetry run ai-hedge-fund-provider login openai-codex"


def login_openai_codex(print_fn=print, prompt_fn=input) -> Any:
    """Reuse a saved Codex login or start an interactive ChatGPT OAuth flow."""
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
    except ImportError as exc:
        raise RuntimeError("OpenAI Codex OAuth requires the optional dependency. " "Run: poetry install -E codex") from exc

    try:
        token = get_token()
    except Exception:
        token = None
    if token and getattr(token, "access", None):
        return token
    return login_oauth_interactive(print_fn=print_fn, prompt_fn=prompt_fn)


def get_openai_codex_login_status() -> Any | None:
    """Return a valid persisted OAuth token without exposing its value."""
    try:
        from oauth_cli_kit import get_token
    except ImportError:
        return None
    try:
        token = get_token()
    except Exception:
        return None
    if token and getattr(token, "access", None) and getattr(token, "account_id", None):
        return token
    return None


def _get_codex_token() -> Any:
    try:
        from oauth_cli_kit import get_token
    except ImportError as exc:
        raise RuntimeError("OpenAI Codex OAuth requires the optional dependency. " "Run: poetry install -E codex") from exc

    try:
        token = get_token()
    except Exception as exc:
        raise RuntimeError(f"OpenAI Codex is not logged in. Run: {LOGIN_COMMAND}") from exc
    if not (token and getattr(token, "access", None) and getattr(token, "account_id", None)):
        raise RuntimeError(f"OpenAI Codex is not logged in. Run: {LOGIN_COMMAND}")
    return token


def validate_codex_base_url(url: str) -> str:
    """Allow OAuth credentials to reach only the known ChatGPT Codex endpoint."""
    value = (url or DEFAULT_CODEX_URL).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "chatgpt.com" or parsed.path != "/backend-api/codex/responses" or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("OpenAI Codex OAuth only supports " "https://chatgpt.com/backend-api/codex/responses")
    return value


def _build_headers(account_id: str, access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": "ai-hedge-fund",
        "User-Agent": "ai-hedge-fund (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _strip_model_prefix(model: str) -> str:
    if model.startswith(("openai-codex/", "openai_codex/")):
        return model.split("/", 1)[1]
    return model


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _coerce_messages(prompt: Any) -> list[Any]:
    if hasattr(prompt, "to_messages"):
        return list(prompt.to_messages())
    if isinstance(prompt, BaseMessage):
        return [prompt]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, (list, tuple)):
        return list(prompt)
    return [{"role": "user", "content": str(prompt)}]


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "user")
    message_type = str(getattr(message, "type", "human"))
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(message_type, message_type)


def _message_content(message: Any) -> Any:
    return message.get("content") if isinstance(message, dict) else getattr(message, "content", "")


def _convert_messages(prompt: Any) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []

    for index, message in enumerate(_coerce_messages(prompt)):
        role = _message_role(message)
        text = _stringify_content(_message_content(message))
        if role == "system":
            if text:
                instructions.append(text)
        elif role == "assistant":
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                        "status": "completed",
                        "id": f"msg_{index}",
                    }
                )
        elif role == "tool":
            call_id = message.get("tool_call_id") if isinstance(message, dict) else getattr(message, "tool_call_id", None)
            if call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call_id).split("|", 1)[0],
                        "output": text,
                    }
                )
        else:
            input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            )

    return "\n\n".join(instructions), input_items


def _events_from_lines(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    buffer: list[str] = []

    def flush() -> dict[str, Any] | None:
        data_lines = [line[5:].strip() for line in buffer if line.startswith("data:")]
        buffer.clear()
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    for line in lines:
        if line == "":
            if buffer:
                event = flush()
                if event is not None:
                    yield event
            continue
        buffer.append(line)
    if buffer:
        event = flush()
        if event is not None:
            yield event


def _text_from_events(events: Iterable[dict[str, Any]]) -> Iterable[str]:
    for event in events:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta") or ""
            if delta:
                yield str(delta)
        elif event_type in {"error", "response.failed"}:
            detail = event.get("error") or event.get("message") or event
            raise RuntimeError(f"OpenAI Codex response failed: {str(detail)[:500]}")


class OpenAICodexLLM:
    """Small LangChain-compatible adapter used by the existing agent helpers."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        timeout: int = 120,
        reasoning_effort: str | None = None,
        codex_url: str | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.codex_url = validate_codex_base_url(codex_url or os.getenv("OPENAI_CODEX_BASE_URL") or DEFAULT_CODEX_URL)

    def _body(self, prompt: Any, *, stream: bool) -> dict[str, Any]:
        instructions, input_items = _convert_messages(prompt)
        cache_source = json.dumps(
            {"instructions": instructions, "input": input_items},
            ensure_ascii=True,
            sort_keys=True,
        )
        body: dict[str, Any] = {
            "model": _strip_model_prefix(self.model),
            "store": False,
            "stream": stream,
            "instructions": instructions,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": hashlib.sha256(cache_source.encode("utf-8")).hexdigest(),
        }
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            body["reasoning"] = {"effort": self.reasoning_effort.lower()}
        return body

    def _headers(self) -> dict[str, str]:
        token = _get_codex_token()
        return _build_headers(str(token.account_id), str(token.access))

    def stream(self, prompt: Any, config: dict[str, Any] | None = None) -> Iterable[AIMessage]:
        timeout = (config or {}).get("timeout") or self.timeout
        with httpx.Client(timeout=timeout, follow_redirects=True, trust_env=True) as client:
            with client.stream(
                "POST",
                self.codex_url,
                headers=self._headers(),
                json=self._body(prompt, stream=True),
            ) as response:
                if response.status_code != 200:
                    raw = response.read().decode("utf-8", "ignore")
                    raise RuntimeError(f"OpenAI Codex HTTP {response.status_code}: {raw[:500]}")
                for text in _text_from_events(_events_from_lines(response.iter_lines())):
                    yield AIMessage(content=text)

    def invoke(self, prompt: Any, config: dict[str, Any] | None = None) -> AIMessage:
        content = "".join(str(chunk.content) for chunk in self.stream(prompt, config=config))
        return AIMessage(content=content)

    async def ainvoke(self, prompt: Any, config: dict[str, Any] | None = None) -> AIMessage:
        return await asyncio.to_thread(self.invoke, prompt, config)
