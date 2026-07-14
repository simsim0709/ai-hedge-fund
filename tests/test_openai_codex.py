"""Tests for the optional ChatGPT/Codex OAuth provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.cli import provider as provider_cli
from src.llm.models import find_model_by_name, get_model, ModelProvider
from src.llm.openai_codex import (
    _events_from_lines,
    _strip_model_prefix,
    _text_from_events,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_URL,
    LOGIN_COMMAND,
    OpenAICodexLLM,
    validate_codex_base_url,
)
from src.utils.llm import call_llm


def test_codex_model_is_registered_without_json_mode() -> None:
    model = find_model_by_name(DEFAULT_CODEX_MODEL)

    assert model is not None
    assert model.provider == ModelProvider.OPENAI_CODEX
    assert model.has_json_mode() is False


def test_get_model_routes_codex_without_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = get_model(DEFAULT_CODEX_MODEL, ModelProvider.OPENAI_CODEX)

    assert isinstance(model, OpenAICodexLLM)


def test_call_llm_uses_codex_json_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Check(BaseModel):
        ok: bool

    fake_model = SimpleNamespace(invoke=lambda prompt: AIMessage(content='{"ok":true}'))
    monkeypatch.setattr("src.utils.llm.get_model", lambda *args, **kwargs: fake_model)
    state = {
        "metadata": {
            "model_name": DEFAULT_CODEX_MODEL,
            "model_provider": ModelProvider.OPENAI_CODEX.value,
        }
    }

    result = call_llm(
        "Return JSON.",
        Check,
        agent_name="test_agent",
        state=state,
        max_retries=1,
    )

    assert result == Check(ok=True)


def test_codex_base_url_is_strictly_allowlisted() -> None:
    assert validate_codex_base_url(DEFAULT_CODEX_URL + "/") == DEFAULT_CODEX_URL

    for unsafe_url in (
        "http://chatgpt.com/backend-api/codex/responses",
        "https://example.com/backend-api/codex/responses",
        "https://chatgpt.com/backend-api/codex/responses?target=other",
        "https://api.openai.com/v1/responses",
    ):
        with pytest.raises(ValueError):
            validate_codex_base_url(unsafe_url)


def test_codex_body_converts_langchain_prompt() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return JSON only."),
            ("human", "Analyze {ticker}."),
        ]
    ).invoke({"ticker": "AAPL"})
    adapter = OpenAICodexLLM(model=DEFAULT_CODEX_MODEL)

    body = adapter._body(prompt, stream=True)

    assert _strip_model_prefix(DEFAULT_CODEX_MODEL) == "gpt-5.4"
    assert body["model"] == "gpt-5.4"
    assert body["instructions"] == "Return JSON only."
    assert body["input"][0]["content"][0]["text"] == "Analyze AAPL."
    assert body["store"] is False
    assert body["stream"] is True


def test_codex_headers_use_oauth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.llm.openai_codex as codex_module

    monkeypatch.setattr(
        codex_module,
        "_get_codex_token",
        lambda: SimpleNamespace(access="oauth-access", account_id="account-123"),
    )

    headers = OpenAICodexLLM()._headers()

    assert headers["Authorization"] == "Bearer oauth-access"
    assert headers["chatgpt-account-id"] == "account-123"
    assert headers["originator"] == "ai-hedge-fund"


def test_missing_codex_token_has_login_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    oauth_cli_kit = pytest.importorskip("oauth_cli_kit")

    def missing_token():
        raise RuntimeError("missing")

    monkeypatch.setattr(oauth_cli_kit, "get_token", missing_token)

    with pytest.raises(RuntimeError, match="ai-hedge-fund-provider login openai-codex"):
        OpenAICodexLLM()._headers()

    assert "login openai-codex" in LOGIN_COMMAND


def test_sse_events_yield_text_and_surface_failures() -> None:
    events = list(
        _events_from_lines(
            [
                'data: {"type":"response.output_text.delta","delta":"{\\"signal\\":\\""}',
                "",
                'data: {"type":"response.output_text.delta","delta":"buy\\"}"}',
                "",
                "data: [DONE]",
                "",
            ]
        )
    )

    assert "".join(_text_from_events(events)) == '{"signal":"buy"}'

    with pytest.raises(RuntimeError, match="response failed"):
        list(_text_from_events([{"type": "response.failed", "message": "denied"}]))


def test_provider_status_command_does_not_print_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        provider_cli,
        "get_openai_codex_login_status",
        lambda: SimpleNamespace(access="secret-value", account_id="account-123"),
    )

    assert provider_cli.main(["status", "openai-codex"]) == 0
    output = capsys.readouterr().out
    assert "ready" in output
    assert "account-123" in output
    assert "secret-value" not in output
