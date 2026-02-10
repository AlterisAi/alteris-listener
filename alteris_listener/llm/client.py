"""LLM client abstraction for running queries against Gemini or Claude."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "alteris-listener"


def _get_api_key(env_var: str, keychain_account: str) -> Optional[str]:
    """Load API key from environment variable or macOS Keychain.

    Checks env var first, then Keychain (set via `alteris-listener set-key`).
    """
    key = os.environ.get(env_var)
    if key:
        return key

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", keychain_account, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return None


class LLMClient:
    """Unified interface for calling Gemini or Claude APIs."""

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        thinking_level: str = "low",
    ):
        self.provider = provider.lower()
        self.thinking_level = thinking_level

        if self.provider == "gemini":
            self.model = model or "gemini-3-flash-preview"
            self._init_gemini()
        elif self.provider == "claude":
            self.model = model or "claude-haiku-4-5-20251001"
            self._init_claude()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'claude'.")

    def _init_gemini(self):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install google-genai")

        api_key = _get_api_key("GEMINI_API_KEY", "gemini")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Either:\n"
                "  • Run: alteris-listener set-key gemini\n"
                "  • Or:  export GEMINI_API_KEY='your-key'"
            )
        self._gemini_client = genai.Client(api_key=api_key)

    def _init_claude(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        api_key = _get_api_key("ANTHROPIC_API_KEY", "claude")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Either:\n"
                "  • Run: alteris-listener set-key claude\n"
                "  • Or:  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self._claude_client = anthropic.Anthropic(api_key=api_key)

    def run(self, system_prompt: str, user_message: str) -> str:
        """Send system + user message to the LLM and return the text response."""
        if self.provider == "gemini":
            return self._run_gemini(system_prompt, user_message)
        return self._run_claude(system_prompt, user_message)

    def _run_gemini(self, system_prompt: str, user_message: str) -> str:
        from google.genai import types

        thinking_budgets = {
            "off": -1,
            "minimal": 128,
            "low": 1024,
            "medium": 4096,
            "high": 16384,
        }
        budget = thinking_budgets.get(self.thinking_level, 1024)

        # Only include thinking_config for models that support it
        # gemini-2.5-* and gemini-3-* support thinking; 2.0 does not
        model_supports_thinking = any(
            self.model.startswith(p) for p in ("gemini-2.5", "gemini-3")
        )

        if model_supports_thinking and self.thinking_level != "off":
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_budget=budget),
                max_output_tokens=16384,
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=16384,
            )

        response = self._gemini_client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=config,
        )

        # Extract text from response parts (skip thinking parts)
        text_parts = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text and not getattr(part, "thought", False):
                    text_parts.append(part.text)

        return "".join(text_parts)

    def _run_claude(self, system_prompt: str, user_message: str) -> str:
        response = self._claude_client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def run_json(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        """Run and parse the response as JSON."""
        raw = self.run(system_prompt, user_message)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            logger.warning("Raw LLM response:\n%s", raw)
            return {"raw_text": raw}
