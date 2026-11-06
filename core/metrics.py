"""Pipeline performance metrics — latency tracking, throughput, and operational counters."""

from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTimer:
    """Tracks wall-clock duration of a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, "duration_ms": round(self.duration_ms, 2)}


@contextmanager
def timed_stage(name: str):
    """Context manager that yields a StageTimer, auto-starting and stopping it.

    Usage::

        with timed_stage("retrieval") as t:
            runbooks = store.search(...)
        print(t.duration_ms)
    """
    timer = StageTimer(name=name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


@dataclass
class PipelineMetrics:
    """Aggregates performance counters across multiple pipeline runs.

    Thread-safe via a simple lock — suitable for the FastAPI and CLI
    contexts where events may be processed concurrently.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Counters
    total_events: int = 0
    recommendations: int = 0
    dropped: int = 0
    skipped: int = 0
    circuit_breaker_trips: int = 0

    # Backend usage counters
    ollama_calls: int = 0
    groq_calls: int = 0
    simulation_calls: int = 0

    # Latency accumulators (milliseconds)
    _parse_latencies: list[float] = field(default_factory=list, repr=False)
    _retrieval_latencies: list[float] = field(default_factory=list, repr=False)
    _inference_latencies: list[float] = field(default_factory=list, repr=False)
    _guardrail_latencies: list[float] = field(default_factory=list, repr=False)
    _e2e_latencies: list[float] = field(default_factory=list, repr=False)

    # ── Recording helpers ────────────────────────────────────────────

    def record_event(
        self,
        status: str,
        backend: str = "",
        parse_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        inference_ms: float = 0.0,
        guardrail_ms: float = 0.0,
        e2e_ms: float = 0.0,
    ) -> None:
        """Record the result of one pipeline run."""
        with self._lock:
            self.total_events += 1

            if status == "recommendation":
                self.recommendations += 1
            elif status == "dropped":
                self.dropped += 1
                self.circuit_breaker_trips += 1
            elif status == "skipped":
                self.skipped += 1

            if backend == "ollama":
                self.ollama_calls += 1
            elif "groq" in backend:
                self.groq_calls += 1
            elif backend == "simulation":
                self.simulation_calls += 1

            if parse_ms > 0:
                self._parse_latencies.append(parse_ms)
            if retrieval_ms > 0:
                self._retrieval_latencies.append(retrieval_ms)
            if inference_ms > 0:
                self._inference_latencies.append(inference_ms)
            if guardrail_ms > 0:
                self._guardrail_latencies.append(guardrail_ms)
            if e2e_ms > 0:
                self._e2e_latencies.append(e2e_ms)

    # ── Statistics ───────────────────────────────────────────────────

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float]:
        """Compute min / avg / max / p95 over a list of float values."""
        if not values:
            return {"min_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "count": 0}
        s = sorted(values)
        n = len(s)
        idx_95 = min(int(n * 0.95), n - 1)
        return {
            "min_ms": round(s[0], 2),
            "avg_ms": round(sum(s) / n, 2),
            "max_ms": round(s[-1], 2),
            "p95_ms": round(s[idx_95], 2),
            "count": n,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            return {
                "counters": {
                    "total_events": self.total_events,
                    "recommendations": self.recommendations,
                    "dropped": self.dropped,
                    "skipped": self.skipped,
                    "circuit_breaker_trips": self.circuit_breaker_trips,
                    "success_rate": round(
                        self.recommendations / self.total_events * 100, 1
                    )
                    if self.total_events
                    else 0.0,
                },
                "backend_usage": {
                    "ollama": self.ollama_calls,
                    "groq": self.groq_calls,
                    "simulation": self.simulation_calls,
                },
                "latency": {
                    "parsing": self._stats(self._parse_latencies),
                    "retrieval": self._stats(self._retrieval_latencies),
                    "inference": self._stats(self._inference_latencies),
                    "guardrail": self._stats(self._guardrail_latencies),
                    "end_to_end": self._stats(self._e2e_latencies),
                },
            }

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        with self._lock:
            self.total_events = 0
            self.recommendations = 0
            self.dropped = 0
            self.skipped = 0
            self.circuit_breaker_trips = 0
            self.ollama_calls = 0
            self.groq_calls = 0
            self.simulation_calls = 0
            self._parse_latencies.clear()
            self._retrieval_latencies.clear()
            self._inference_latencies.clear()
            self._guardrail_latencies.clear()
            self._e2e_latencies.clear()

