"""Circuit Breaker and command safety guardrails."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from typing import Any

MIN_CONFIDENCE_DEFAULT = 0.75
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


@dataclass
class CircuitBreaker:
    minimum_confidence: float = MIN_CONFIDENCE_DEFAULT
    is_open: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        """Trip the circuit breaker, halting further automated actions."""
        self.is_open = True
        self.reason = reason

    def reset(self) -> None:
        """Reset the breaker to healthy/closed state after human review."""
        self.is_open = False
        self.reason = ""

    def validate(self, recommendation: dict[str, Any]) -> str | None:
        """Validate an LLM recommendation against schema and safety guardrails.
        
        Returns None if safe, or an error reason string if rejected (which trips the breaker).
        """
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

