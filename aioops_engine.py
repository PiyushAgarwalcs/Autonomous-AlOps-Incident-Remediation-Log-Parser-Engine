#!/usr/bin/env python3
"""Local log monitoring, ChromaDB runbook retrieval, and guarded Ollama recommendations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

# Core module imports
from core.guardrails import (
    CONFIDENCE_FLAGS,
    HIGH_RISK_COMMANDS,
    HIGH_RISK_PATTERN,
    MIN_CONFIDENCE_DEFAULT,
    SHELL_OPERATORS,
    CircuitBreaker,
)
from core.llm_client import (
    CHAT_MODEL_DEFAULT,
    EMBEDDING_MODEL_DEFAULT,
    OLLAMA_DEFAULT,
    RESPONSE_SCHEMA,
    OllamaClient,
)
from core.parser import (
    ERROR_LEVELS,
    ERROR_WORD,
    follow_file,
    parse_log_line,
)
from core.vector_store import (
    ChromaRunbooks,
    load_runbooks,
)


def emit(record: dict[str, Any]) -> None:
    """Emit structured JSON record to standard output."""
    print(json.dumps(record, ensure_ascii=False), flush=True)


logger = logging.getLogger(__name__)


def process_event(
    event: dict[str, str],
    store: ChromaRunbooks,
    ollama: OllamaClient,
    breaker: CircuitBreaker,
    top_k: int = 3,
) -> dict[str, Any]:
    """Process a single parsed error event through retrieval, inference, and guardrails."""
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
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=Path("server.log"))
    parser.add_argument("--runbooks", type=Path, default=Path("runbooks.jsonl"))
    parser.add_argument("--once", action="store_true", help="Process current file contents and exit.")
    parser.add_argument("--from-end", action="store_true", help="Ignore existing lines when following.")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sync-runbooks", action="store_true", help="Sync runbooks.jsonl to local ChromaDB.")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="Local directory for ChromaDB storage.")
    parser.add_argument("--collection-name", default="sre_runbooks", help="ChromaDB collection name.")
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ollama = OllamaClient(
        args.ollama_url,
        args.chat_model,
        args.embedding_model,
    )
    store = ChromaRunbooks(persist_dir=args.chroma_dir, collection_name=args.collection_name)

    if args.sync_runbooks:
        count = store.upsert(load_runbooks(args.runbooks), ollama)
        emit({
            "status": "runbooks_indexed",
            "count": count,
            "engine": "chromadb" if store.is_using_chroma else "local_vector_store",
            "persist_dir": args.chroma_dir,
        })
        logger.info("Indexed %d runbooks into %s.", count, args.chroma_dir)
        return 0

    breaker = CircuitBreaker(minimum_confidence=args.min_confidence)
    lines: Iterator[str]
    if args.once:
        with args.log_file.open("r", encoding="utf-8", errors="replace") as source:
            lines = iter(source.readlines())
    else:
        logger.info("Following %s (poll_interval=%.1fs, from_end=%s).", args.log_file, args.poll_interval, args.from_end)
        lines = follow_file(args.log_file, args.poll_interval, args.from_end)

    try:
        for line in lines:
            event = parse_log_line(line)
            if event:
                result = process_event(event, store, ollama, breaker, args.top_k)
                emit(result)
                logger.info("Processed event [%s/%s] → status=%s", event["service"], event["level"], result["status"])
    except KeyboardInterrupt:
        logger.info("Interrupted by operator — shutting down cleanly.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
