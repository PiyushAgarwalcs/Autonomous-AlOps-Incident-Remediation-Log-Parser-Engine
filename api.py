"""FastAPI Webhook & REST Ingestion API for external backends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.guardrails import CircuitBreaker
from core.llm_client import OllamaClient
from core.groq_client import GroqClient
from core.parser import parse_log_line
from core.vector_store import ChromaRunbooks, load_runbooks
from core.metrics import PipelineMetrics, timed_stage

app = FastAPI(
    title="AIOps Incident Remediation & Ingestion API",
    description="Webhook endpoint for external services to push logs and receive SRE remediation plans.",
    version="2.0.0",
)

# Shared in-memory/file buffer for received incidents
INGESTED_LOGS_FILE = Path("ingested_logs.jsonl")
RUNBOOKS_FILE = Path("runbooks.jsonl")

# Initialize components
breaker = CircuitBreaker()
vector_store = ChromaRunbooks(persist_dir="./chroma_db", collection_name="sre_runbooks")
ollama = OllamaClient()
groq = GroqClient(api_key=os.getenv("GROQ_API_KEY", ""))
metrics = PipelineMetrics()


class LogPayload(BaseModel):
    """Payload format for external service log pushes."""
    timestamp: str | None = None
    level: str = Field(..., description="ERROR, CRITICAL, or FATAL")
    service: str = Field(default="external-service", description="Source service name")
    error_code: str | None = None
    message: str = Field(..., description="The error log message")


class RawTextPayload(BaseModel):
    """Plain text log payload."""
    log_text: str


@app.get("/health")
def health_check():
    """Service health and circuit breaker status."""
    return {
        "status": "healthy",
        "circuit_breaker": "OPEN (TRIPPED)" if breaker.is_open else "CLOSED (HEALTHY)",
        "circuit_breaker_reason": breaker.reason if breaker.is_open else None,
        "ollama_online": ollama.is_alive(),
        "groq_configured": groq.is_configured(),
    }


@app.get("/metrics")
def get_metrics():
    """Return pipeline performance metrics snapshot."""
    return metrics.snapshot()


@app.post("/metrics/reset")
def reset_metrics():
    """Reset all accumulated pipeline metrics."""
    metrics.reset()
    return {"status": "metrics_reset"}


@app.post("/ingest")
def ingest_log(payload: LogPayload):
    """Ingest a structured error log from any external backend."""
    raw_line = payload.model_dump_json()
    event = parse_log_line(raw_line)

    if not event:
        raise HTTPException(
            status_code=400,
            detail="Log is not an actionable error. Severity must be ERROR, CRITICAL, or FATAL.",
        )

    # Persist to ingested_logs.jsonl for the dashboard to display
    with INGESTED_LOGS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return {
        "status": "accepted",
        "message": "Incident received and queued for remediation analysis.",
        "event": event,
    }


@app.post("/ingest/raw")
def ingest_raw_log(payload: RawTextPayload):
    """Ingest a raw syslog string (e.g., from journalctl, stdout)."""
    event = parse_log_line(payload.log_text)
    if not event:
        raise HTTPException(
            status_code=400,
            detail="Log line contains no actionable error level (ERROR, CRITICAL, FATAL).",
        )

    with INGESTED_LOGS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return {
        "status": "accepted",
        "event": event,
    }


@app.get("/incidents")
def list_incidents(limit: int = 50):
    """Retrieve recently ingested incidents."""
    if not INGESTED_LOGS_FILE.exists():
        return {"incidents": []}

    incidents = []
    with INGESTED_LOGS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    incidents.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"incidents": incidents[-limit:]}


@app.post("/remediate")
def remediate_incident(payload: LogPayload):
    """Directly analyze an error and return a validated SRE remediation plan."""
    with timed_stage("end_to_end") as e2e:
        with timed_stage("parsing") as t_parse:
            event = parse_log_line(payload.model_dump_json())
        if not event:
            raise HTTPException(status_code=400, detail="Invalid error event.")

        if breaker.is_open:
            metrics.record_event("dropped", parse_ms=t_parse.duration_ms, e2e_ms=e2e.duration_ms)
            return {
                "status": "blocked",
                "reason": f"Circuit breaker is tripped: {breaker.reason}",
                "human_approval_required": True,
                "executed": False,
            }

        # Retrieve runbook
        query = f"{event['error_code']} {event['service']} {event['message']}"
        runbooks = []
        with timed_stage("retrieval") as t_ret:
            if RUNBOOKS_FILE.exists():
                if ollama.is_alive():
                    try:
                        runbooks = vector_store.search(query, ollama, top_k=1)
                    except Exception:
                        runbooks = []

                if not runbooks:
                    raw_runbooks = load_runbooks(RUNBOOKS_FILE)
                    err_code = event.get("error_code", "").upper()
                    msg_lower = event.get("message", "").lower()
                    for rb in raw_runbooks:
                        rb_code = rb.get("error_code", "").upper()
                        if rb_code and rb_code in err_code:
                            runbooks = [{**rb, "score": 0.95}]
                            break
                        keywords = [k for k in rb_code.lower().split("_") if len(k) > 2]
                        if any(kw in msg_lower for kw in keywords):
                            runbooks = [{**rb, "score": 0.91}]
                            break

            if not runbooks:
                runbooks = [{
                    "id": "generic-incident",
                    "title": "General Service Triage",
                    "text": "Inspect the service logs and check health status before changing settings.",
                }]

        # Inference (Local Ollama, Groq cloud, or simulation fallback)
        rec = None
        backend_used = "simulation"
        with timed_stage("inference") as t_inf:
            if ollama.is_alive():
                try:
                    rec = ollama.recommend(event, runbooks)
                    backend_used = "ollama"
                except Exception:
                    rec = None

            if not rec and groq.is_configured():
                try:
                    rec = groq.recommend(event, runbooks)
                    backend_used = "groq"
                except Exception:
                    rec = None

            if not rec:
                backend_used = "simulation"
                rec = {
                    "summary": f"Check status of {event['service']} and inspect recent system logs.",
                    "command": f"systemctl status {event['service']}.service",
                    "confidence": 0.93,
                    "confidence_flags": {
                        "diagnosis": True,
                        "runbook_support": True,
                        "command_safety": True,
                    },
                }

        with timed_stage("guardrail") as t_guard:
            rejection = breaker.validate(rec)

        if rejection:
            metrics.record_event(
                "dropped", backend=backend_used,
                parse_ms=t_parse.duration_ms, retrieval_ms=t_ret.duration_ms,
                inference_ms=t_inf.duration_ms, guardrail_ms=t_guard.duration_ms,
                e2e_ms=e2e.duration_ms,
            )
            return {
                "status": "dropped",
                "reason": rejection,
                "human_approval_required": True,
                "executed": False,
            }

    latency = {
        "parsing_ms": round(t_parse.duration_ms, 2),
        "retrieval_ms": round(t_ret.duration_ms, 2),
        "inference_ms": round(t_inf.duration_ms, 2),
        "guardrail_ms": round(t_guard.duration_ms, 2),
        "end_to_end_ms": round(e2e.duration_ms, 2),
    }

    metrics.record_event(
        "recommendation", backend=backend_used,
        parse_ms=t_parse.duration_ms, retrieval_ms=t_ret.duration_ms,
        inference_ms=t_inf.duration_ms, guardrail_ms=t_guard.duration_ms,
        e2e_ms=e2e.duration_ms,
    )

    return {
        "status": "recommendation",
        "event": event,
        "runbook": runbooks[0].get("id", ""),
        "summary": rec["summary"],
        "command": rec["command"],
        "confidence": rec["confidence"],
        "backend": backend_used,
        "human_approval_required": True,
        "executed": False,
        "latency": latency,
    }

