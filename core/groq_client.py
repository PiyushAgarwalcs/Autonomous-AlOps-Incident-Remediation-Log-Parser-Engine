"""Groq Cloud LLM client for live inference on deployed environments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_DEFAULT = "llama-3.1-8b-instant"


class GroqClient:
    """Client for Groq Cloud LLM API with OpenAI-compatible JSON schema output."""

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_MODEL_DEFAULT,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Check if API key is provided and non-empty."""
        return bool(self.api_key and self.api_key.startswith("gsk_"))

    def recommend(self, event: dict[str, str], runbooks: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate structured remediation recommendation using Groq LLM."""
        if not self.is_configured():
            raise ValueError("Groq API key is not configured.")

        system_prompt = (
            "You are a cautious SRE assistant. Never claim an action ran. "
            "You must respond with ONLY a valid JSON object with the following exact keys:\n"
            "{\n"
            '  "summary": "Short explanation of the diagnosis and required read-only check.",\n'
            '  "command": "Single safe bash command for an operator to review (e.g., systemctl status ...). If evidence is insufficient or risky, use an empty string.",\n'
            '  "confidence": 0.95,\n'
            '  "confidence_flags": {\n'
            '    "diagnosis": true,\n'
            '    "runbook_support": true,\n'
            '    "command_safety": true\n'
            "  }\n"
            "}\n"
            "Do not use shell chaining (;, &&, |), redirection, or high-risk destructive commands (rm, chmod, sudo, dd, kill)."
        )

        user_prompt = (
            f"Error event:\n{json.dumps(event, ensure_ascii=False)}\n\n"
            f"Retrieved SRE Runbooks:\n{json.dumps(runbooks, ensure_ascii=False)}\n\n"
            "Analyze the error using the runbooks and return the JSON recommendation."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 500,
        }

        request = urllib.request.Request(
            GROQ_API_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AIOps-Engine/2.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq API HTTP error {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Groq network connection failed: {error}") from error

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices in response.")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Groq returned empty message content.")

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ValueError(f"Groq response was not valid JSON: {content}") from error

        if not isinstance(parsed, dict):
            raise ValueError("Groq response must be a JSON object.")

        return parsed
