"""Local Ollama client for embeddings and structured remediation recommendations."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

OLLAMA_DEFAULT = "http://localhost:11434"
CHAT_MODEL_DEFAULT = "qwen2.5:7b"
EMBEDDING_MODEL_DEFAULT = "nomic-embed-text"

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "command": {"type": "string"},
        "confidence": {"type": "number"},
        "confidence_flags": {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "boolean"},
                "runbook_support": {"type": "boolean"},
                "command_safety": {"type": "boolean"},
            },
            "required": ["command_safety", "diagnosis", "runbook_support"],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "command", "confidence", "confidence_flags"],
    "additionalProperties": False,
}


class OllamaClient:
    """Interface to local Ollama inference server."""

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT,
        chat_model: str = CHAT_MODEL_DEFAULT,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT,
        timeout: float = 120.0,
    ) -> None:
        parsed_url = urllib.parse.urlsplit(base_url)
        if parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must be configured with a localhost URL.")
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Ollama URL must be an HTTP(S) localhost URL.")

        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Local Ollama request failed: {error}") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected response from Ollama {endpoint}.")
        return result

    def is_alive(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=2.0) as response:
                return response.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        """Generate text embedding vector using local model."""
        result = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": text},
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
            raise RuntimeError("Ollama returned no text embedding.")
        return [float(value) for value in embeddings[0]]

    def recommend(self, event: dict[str, str], runbooks: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask model for structured remediation recommendation based on retrieved runbooks."""
        prompt = (
            "Analyze this server error using only the supplied SRE runbooks. "
            "Return one JSON object matching the required schema. Provide a single bash "
            "command for an operator to review; do not execute it. Prefer read-only "
            "diagnostics. If evidence is insufficient, use an empty command, low confidence, "
            "and false confidence flags. Do not use shell chaining, pipes, or redirection.\n\n"
            f"Error event:\n{json.dumps(event, ensure_ascii=False)}\n\n"
            f"Retrieved runbooks:\n{json.dumps(runbooks, ensure_ascii=False)}"
        )
        result = self._post(
            "/api/chat",
            {
                "model": self.chat_model,
                "stream": False,
                "format": RESPONSE_SCHEMA,
                "options": {"temperature": 0.1},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cautious SRE assistant. Never claim an action ran.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned no chat content.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Ollama response was not valid JSON.") from error
        if not isinstance(parsed, dict):
            raise ValueError("Ollama response must be a JSON object.")
        return parsed

