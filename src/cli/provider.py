"""Authentication commands for optional LLM providers."""

from __future__ import annotations

import argparse

from src.llm.openai_codex import get_openai_codex_login_status, login_openai_codex


def _require_codex_provider(provider: str) -> None:
    if provider.strip().lower().replace("_", "-") != "openai-codex":
        raise ValueError("Unknown OAuth provider. Supported: openai-codex")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage optional LLM provider authentication")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("login", "status"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("provider", help="Provider name (openai-codex)")

    args = parser.parse_args(argv)
    try:
        _require_codex_provider(args.provider)
        if args.command == "login":
            token = login_openai_codex()
            account = getattr(token, "account_id", None) or "ChatGPT"
            print(f"Authenticated with OpenAI Codex ({account})")
            return 0

        token = get_openai_codex_login_status()
        if not token:
            print("OpenAI Codex OAuth is not configured")
            return 1
        account = getattr(token, "account_id", None) or "ChatGPT"
        print(f"OpenAI Codex OAuth is ready ({account})")
        return 0
    except Exception as exc:
        print(f"Authentication error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
