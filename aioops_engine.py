#!/usr/bin/env python3
"""Local log monitoring, Pinecone runbook retrieval, and guarded Ollama recommendations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


OLLAMA_DEFAULT = "http://localhost:11434"
CHAT_MODEL_DEFAULT = "qwen2.5:7b"
EMBEDDING_MODEL_DEFAULT = "nomic-embed-text"
MIN_CONFIDENCE_DEFAULT = 0.75
ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}
CONFIDENCE_FLAGS = {"diagnosis", "runbook_support", "command_safety"}
HIGH_RISK_COMMANDS = {
    "rm",
    "chmod",
    "chown",
    "sudo",
    "doas",
    "su",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
    "kill",
    "pkill",
    "iptables",
    "nft",
    "userdel",
    "usermod",
    "fdisk",
    "parted",
    "wipefs",
    "eval",
    "exec",
}
SHELL_OPERATORS = re.compile(r"[;&|<>`\r\n]|\$\s*\(")
HIGH_RISK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    + "|".join(re.escape(command) for command in sorted(HIGH_RISK_COMMANDS))
    + r")(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
ERROR_WORD = re.compile(r"\b(ERROR|CRITICAL|FATAL)\b", re.IGNORECASE)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "command": {"type": "string"},
        "confidence": {"type": "number"},
        "confidence_flags": {
            "type": "object",
            "properties": {flag: {"type": "boolean"} for flag in sorted(CONFIDENCE_FLAGS)},
            "required": sorted(CONFIDENCE_FLAGS),
            "additionalProperties": False,
        },
    },
    "required": ["summary", "command", "confidence", "confidence_flags"],
    "additionalProperties": False,
}


def parse_log_line(line: str) -> dict[str, str] | None:
    """Parse JSONL or raw text and return only error-level events."""
    text = line.strip()
    if not text:
        return None

    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        match = ERROR_WORD.search(text)
        if not match:
            return None
        record = {"level": match.group(1), "message": text}

    if not isinstance(record, dict):
        return None

    level = str(record.get("level", "")).upper()
    message = record.get("message", record.get("error", record.get("msg", "")))
    if level not in ERROR_LEVELS or not isinstance(message, str) or not message.strip():
        return None

    return {
        "timestamp": str(record.get("timestamp", "")),
        "level": level,
        "service": str(record.get("service", "unknown")),
        "error_code": str(record.get("error_code", record.get("code", ""))),
        "message": message.strip(),
    }


def follow_file(path: Path, poll_interval: float, from_end: bool = False) -> Iterator[str]:
    """Yield appended lines from a local file, polling without extra watcher dependencies."""
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        if from_end:
            log_file.seek(0, os.SEEK_END)
        while True:
            line = log_file.readline()
            if line:
                yield line
            else:
                time.sleep(poll_interval)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
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

    def embed(self, text: str) -> list[float]:
        result = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": text},
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
            raise RuntimeError("Ollama returned no text embedding.")
        return [float(value) for value in embeddings[0]]

    def recommend(self, event: dict[str, str], runbooks: list[dict[str, Any]]) -> dict[str, Any]:
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


class PineconeRunbooks:
    def __init__(self, api_key: str, index_name: str) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as error:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt`.") from error

        self.index = Pinecone(api_key=api_key).Index(index_name)

    def upsert(self, runbooks: list[dict[str, str]], ollama: OllamaClient) -> int:
        vectors = []
        for runbook in runbooks:
            searchable_text = f"{runbook['error_code']} {runbook['title']}\n{runbook['text']}"
            vectors.append(
                {
                    "id": runbook["id"],
                    "values": ollama.embed(searchable_text),
                    "metadata": {
                        "error_code": runbook["error_code"],
                        "title": runbook["title"],
                        "text": runbook["text"],
                    },
                }
            )
        if vectors:
            self.index.upsert(vectors=vectors)
        return len(vectors)

    def search(self, query: str, ollama: OllamaClient, top_k: int) -> list[dict[str, Any]]:
        result = self.index.query(
            vector=ollama.embed(query),
            top_k=top_k,
            include_metadata=True,
        )
        matches = self._field(result, "matches", [])
        if not isinstance(matches, (list, tuple)):
            raise RuntimeError("Pinecone returned an invalid matches response.")
        runbooks = []
        for match in matches:
            metadata = self._field(match, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                raise RuntimeError("Pinecone returned invalid runbook metadata.")
            runbooks.append(
                {
                    "id": self._field(match, "id", ""),
                    "score": self._field(match, "score"),
                    "error_code": metadata.get("error_code", ""),
                    "title": metadata.get("title", ""),
                    "text": metadata.get("text", ""),
                }
            )
        return runbooks

    @staticmethod
    def _field(value: Any, name: str, default: Any) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)


def load_runbooks(path: Path) -> list[dict[str, str]]:
    runbooks = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid runbook JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Runbook on line {line_number} must be a JSON object.")
            required = ("id", "error_code", "title", "text")
            if any(not isinstance(record.get(key), str) or not record[key].strip() for key in required):
                raise ValueError(f"Runbook on line {line_number} needs non-empty id/error_code/title/text.")
            runbooks.append({key: record[key].strip() for key in required})
    return runbooks


@dataclass
class CircuitBreaker:
    minimum_confidence: float = MIN_CONFIDENCE_DEFAULT
    is_open: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        self.is_open = True
        self.reason = reason

    def validate(self, recommendation: dict[str, Any]) -> str | None:
        if self.is_open:
            return f"Circuit breaker is open: {self.reason}"

        command = recommendation.get("command")
        confidence = recommendation.get("confidence")
        flags = recommendation.get("confidence_flags")
        summary = recommendation.get("summary")
        if (
            not isinstance(command, str)
            or not command.strip()
            or not isinstance(summary, str)
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
            or not math.isfinite(confidence)
            or not isinstance(flags, dict)
            or set(flags) != CONFIDENCE_FLAGS
            or any(not isinstance(value, bool) for value in flags.values())
        ):
            return self._reject("Model response did not satisfy the recommendation schema.")

        if confidence < self.minimum_confidence or not all(flags.values()):
            return self._reject("Model confidence or confidence flags were below the safety threshold.")

        if SHELL_OPERATORS.search(command):
            return self._reject("Shell operators or multiline commands are not allowed.")

        try:
            tokens = shlex.split(command)
        except ValueError:
            return self._reject("Command could not be parsed safely.")
        if not tokens:
            return self._reject("Model returned an empty command.")

        for token in tokens:
            risky_match = HIGH_RISK_PATTERN.search(token)
            if risky_match:
                return self._reject(f"High-risk command keyword blocked: {risky_match.group(0)}.")
        risky_match = HIGH_RISK_PATTERN.search(command)
        if risky_match:
            return self._reject(f"High-risk command keyword blocked: {risky_match.group(0)}.")
        return None

    def _reject(self, reason: str) -> str:
        self.trip(reason)
        return reason


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True)


def process_event(
    event: dict[str, str],
    store: PineconeRunbooks,
    ollama: OllamaClient,
    breaker: CircuitBreaker,
    top_k: int,
) -> dict[str, Any]:
    if breaker.is_open:
        return {"status": "dropped", "reason": f"Circuit breaker is open: {breaker.reason}"}

    runbooks = store.search(
        f"{event['error_code']} {event['service']} {event['message']}",
        ollama,
        top_k,
    )
    if not runbooks:
        return {"status": "skipped", "reason": "No matching runbook found."}

    try:
        recommendation = ollama.recommend(event, runbooks)
    except ValueError as error:
        breaker.trip(str(error))
        return {"status": "dropped", "reason": str(error)}

    rejection = breaker.validate(recommendation)
    if rejection:
        return {"status": "dropped", "reason": rejection}

    return {
        "status": "recommendation",
        "event": event,
        "runbook": runbooks[0].get("id", ""),
        "summary": recommendation["summary"],
        "command": recommendation["command"],
        "confidence": recommendation["confidence"],
        "human_approval_required": True,
        "executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=Path("server.log"))
    parser.add_argument("--runbooks", type=Path, default=Path("runbooks.jsonl"))
    parser.add_argument("--once", action="store_true", help="Process current file contents and exit.")
    parser.add_argument("--from-end", action="store_true", help="Ignore existing lines when following.")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sync-runbooks", action="store_true")
    parser.add_argument("--ollama-url", default=OLLAMA_DEFAULT)
    parser.add_argument("--chat-model", default=CHAT_MODEL_DEFAULT)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_interval <= 0 or args.top_k < 1:
        raise ValueError("--poll-interval must be positive and --top-k must be at least 1.")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1.")

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX")
    if not api_key or not index_name:
        raise RuntimeError("Set PINECONE_API_KEY and PINECONE_INDEX before starting the engine.")

    ollama = OllamaClient(
        args.ollama_url,
        args.chat_model,
        args.embedding_model,
    )
    store = PineconeRunbooks(api_key, index_name)

    if args.sync_runbooks:
        count = store.upsert(load_runbooks(args.runbooks), ollama)
        emit({"status": "runbooks_indexed", "count": count, "index": index_name})
        return 0

    breaker = CircuitBreaker(minimum_confidence=args.min_confidence)
    lines: Iterator[str]
    if args.once:
        with args.log_file.open("r", encoding="utf-8", errors="replace") as source:
            lines = iter(source.readlines())
    else:
        lines = follow_file(args.log_file, args.poll_interval, args.from_end)

    try:
        for line in lines:
            event = parse_log_line(line)
            if event:
                emit(process_event(event, store, ollama, breaker, args.top_k))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
