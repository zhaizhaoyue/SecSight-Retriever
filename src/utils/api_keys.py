from __future__ import annotations

from getpass import getpass
from typing import Optional


def _read_secret(prompt_text: str) -> str:
    """
    Read a secret from stdin, falling back to a normal input prompt when getpass
    is not supported (e.g., inside some IDE consoles).
    """
    try:
        return getpass(prompt_text)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        # Some environments (notebooks) do not support getpass; fall back.
        return input(prompt_text)


def prompt_for_api_key(
    label: str = "LLM",
    *,
    required: bool = True,
    allow_retry: bool = True,
) -> Optional[str]:
    """
    Request an API key from the user without echoing input.

    Args:
        label: Human readable target, e.g., "DeepSeek".
        required: When True, keep prompting until a value is provided.
        allow_retry: When False, only one prompt is attempted.

    Returns:
        The stripped API key or None if the user opted out.
    """
    prompt = f"Enter {label} API key: " if label else "Enter API key: "
    optional_hint = "" if required else " (leave blank to skip)"
    prompt = prompt.rstrip(": ") + optional_hint + ": "

    while True:
        try:
            value = _read_secret(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAPI key entry cancelled.")
            return None

        if value:
            return value
        if not required:
            return None
        if not allow_retry:
            return None
        print("A non-empty API key is required to continue. Please try again.")

