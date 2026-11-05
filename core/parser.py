"""Log parsing and file tailing utilities."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

ERROR_LEVELS = {"ERROR", "CRITICAL", "FATAL"}
ERROR_WORD = re.compile(r"\b(ERROR|CRITICAL|FATAL)\b", re.IGNORECASE)


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
    """Yield appended lines from a local file, polling without extra watcher dependencies.

    Handles log rotation (rename + recreate, or truncation) automatically.
    """
    inode: int | None = None

    def _open_file():
        nonlocal inode
        fh = path.open("r", encoding="utf-8", errors="replace")
        try:
            inode = os.stat(path).st_ino
        except OSError:
            inode = None
        return fh

    log_file = _open_file()
    try:
        if from_end:
            log_file.seek(0, os.SEEK_END)

        while True:
            line = log_file.readline()
            if line:
                yield line
            else:
                time.sleep(poll_interval)

                # Detect log rotation: file replaced (new inode) or truncated
                try:
                    current_stat = os.stat(path)
                    current_inode = current_stat.st_ino
                    current_size = current_stat.st_size
                    file_pos = log_file.tell()
                except OSError:
                    # File temporarily missing during rotation — keep waiting
                    continue

                rotated = (inode is not None and current_inode != inode) or (current_size < file_pos)
                if rotated:
                    logger.info("Log rotation detected for %s — reopening.", path)
                    log_file.close()
                    log_file = _open_file()
    finally:
        log_file.close()

