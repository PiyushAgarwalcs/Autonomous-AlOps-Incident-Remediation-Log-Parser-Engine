"""Interactive Streamlit Dashboard for Autonomous AIOps Incident Remediation & Log Parser Engine."""

import json
import logging
import os
from pathlib import Path
import streamlit as st

from core.guardrails import (
    CONFIDENCE_FLAGS,
    HIGH_RISK_COMMANDS,
    CircuitBreaker,
)
from core.llm_client import (
    CHAT_MODEL_DEFAULT,
    EMBEDDING_MODEL_DEFAULT,
    OLLAMA_DEFAULT,
    OllamaClient,
)
from core.groq_client import (
    GROQ_MODEL_DEFAULT,
    GroqClient,
)
from core.parser import parse_log_line
from core.vector_store import ChromaRunbooks, load_runbooks

# Set page configuration
st.set_page_config(
    page_title="AIOps Remediation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished SRE look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 0.6rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .badge-safe {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 0.25rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 0.25rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #78350f;
        color: #fde68a;
        padding: 0.25rem 0.6rem;
        border-radius: 0.4rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RUNBOOKS_FILE = Path("runbooks.jsonl")
INGESTED_LOGS_FILE = Path("ingested_logs.jsonl")
AUDIT_LOG_FILE = Path("audit_log.jsonl")

logger = logging.getLogger(__name__)


def _load_persisted_audit_log() -> list[dict]:
    """Load audit log entries from the persistent JSONL file."""
    if not AUDIT_LOG_FILE.exists():
        return []
    entries = []
    with AUDIT_LOG_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _append_audit_entry(entry: dict) -> None:
    """Persist a single audit log entry to disk."""
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Initialize Session State
if "breaker" not in st.session_state:
    st.session_state.breaker = CircuitBreaker(minimum_confidence=0.75)

if "audit_log" not in st.session_state:
    # Restore from disk so the audit trail survives page refreshes
    st.session_state.audit_log = _load_persisted_audit_log()


@st.cache_resource
def get_vector_store():
    store = ChromaRunbooks(persist_dir="./chroma_db", collection_name="sre_runbooks")
    return store


vector_store = get_vector_store()

# Read Groq API Key securely from Streamlit secrets or environment variable
configured_groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# Sidebar: Controls & System Health
with st.sidebar:
    st.title("🛡️ Engine Controls")

    st.subheader("1. LLM Backend Status")

    ollama_url = st.text_input("Local Ollama Endpoint", value=OLLAMA_DEFAULT)
    ollama = OllamaClient(
        base_url=ollama_url,
        chat_model=CHAT_MODEL_DEFAULT,
        embedding_model=EMBEDDING_MODEL_DEFAULT,
    )
    ollama_online = ollama.is_alive()

    # Optional custom Groq key if visitor wants to test their own
    custom_groq_key = st.text_input(
        "Groq API Key (Optional Override)",
        type="password",
        help="Leave blank to use the secure server-configured key",
    )
    active_groq_key = custom_groq_key.strip() if custom_groq_key.strip() else configured_groq_key
    groq = GroqClient(api_key=active_groq_key, model=GROQ_MODEL_DEFAULT)

    # Status indicators
    if ollama_online:
        st.success("🟢 Local Ollama Online (100% Air-Gapped)")
        active_backend = "ollama"
    elif groq.is_configured():
        st.success(f"🟢 Groq Cloud LLM Active (`{groq.model}`)")
        active_backend = "groq"
    else:
        st.warning("🟡 Simulation Mode Active")
        active_backend = "simulation"

    st.divider()

    st.subheader("2. Circuit Breaker Guardrail")
    breaker: CircuitBreaker = st.session_state.breaker

    if breaker.is_open:
        st.error(f"🔴 **STATE: TRIPPED / OPEN**\n\n**Reason:** {breaker.reason}")
        if st.button("🔄 Reset Circuit Breaker", type="primary", use_container_width=True):
            breaker.reset()
            st.rerun()
    else:
        st.success("🟢 **STATE: HEALTHY / CLOSED**\n\nMonitoring safe recommendations")

    min_confidence = st.slider(
        "Min Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.05,
    )
    breaker.minimum_confidence = min_confidence

    st.divider()
    st.caption("AIOps SRE Engine v2.0 • Air-Gapped + Cloud Ready")


# Header
st.title("Autonomous AIOps Incident Remediation & Log Parser Engine")
st.markdown(
    "Air-gapped, human-in-the-loop SRE assistant powered by **ChromaDB** and local **Ollama** / **Groq Cloud**. "
    "Features a deterministic **Circuit Breaker** to prevent unauthorized or dangerous shell commands."
)

tab_simulator, tab_uploader, tab_connect, tab_runbooks, tab_guardrails, tab_audit = st.tabs(
    [
        "🎮 Incident Simulator",
        "📂 Upload Log File",
        "🔌 Connect External Backend",
        "📚 SRE Runbooks",
        "🛡️ Guardrail Architecture",
        "📋 Operator Audit Log",
    ]
)


def analyze_event(event: dict[str, str]) -> dict:
    """Execute the retrieval, inference, and guardrail validation for an error event."""
    query_text = f"{event.get('error_code', '')} {event.get('service', '')} {event.get('message', '')}"

    runbooks = []
    if RUNBOOKS_FILE.exists():
        raw_runbooks = load_runbooks(RUNBOOKS_FILE)
        if ollama_online:
            try:
                vector_store.upsert(raw_runbooks, ollama)
                runbooks = vector_store.search(query_text, ollama, top_k=1)
            except Exception:
                runbooks = []

        if not runbooks:
            # Semantic keyword matching over loaded runbooks
            err_code = event.get("error_code", "").upper()
            msg_lower = event.get("message", "").lower()
            for rb in raw_runbooks:
                rb_code = rb.get("error_code", "").upper()
                if rb_code and rb_code in err_code:
                    runbooks = [{**rb, "score": 0.96}]
                    break
                keywords = [k for k in rb_code.lower().split("_") if len(k) > 2]
                if any(kw in msg_lower for kw in keywords):
                    runbooks = [{**rb, "score": 0.92}]
                    break

    if not runbooks:
        runbooks = [
            {
                "id": "generic-triage",
                "score": 0.85,
                "error_code": event.get("error_code", "GENERIC_ERROR"),
                "title": "General Service Triage",
                "text": "Inspect the service logs and check health status before changing settings.",
            }
        ]

    # Step 2: LLM Inference
    rec = None
    backend_used = active_backend

    if active_backend == "ollama":
        try:
            rec = ollama.recommend(event, runbooks)
        except Exception:
            backend_used = "groq"

    if not rec and groq.is_configured():
        try:
            rec = groq.recommend(event, runbooks)
            backend_used = f"groq ({groq.model})"
        except Exception:
            backend_used = "simulation"

    if not rec:
        backend_used = "simulation"
        msg = event.get("message", "").lower()
        err_code = event.get("error_code", "")
        if "rm -rf" in msg or "SECURITY_ALERT" in err_code or "wipe" in msg:
            rec = {
                "summary": "Attempting cleanup of temporary directory.",
                "command": "rm -rf /tmp/service",
                "confidence": 0.95,
                "confidence_flags": {
                    "diagnosis": True,
                    "runbook_support": False,
                    "command_safety": False,
                },
            }
        elif "oom" in msg or "OOM_KILL" in err_code:
            rec = {
                "summary": "Process terminated by kernel OOM killer. Check memory usage and journal logs.",
                "command": "journalctl -k | grep -i oom",
                "confidence": 0.94,
                "confidence_flags": {
                    "diagnosis": True,
                    "runbook_support": True,
                    "command_safety": True,
                },
            }
        elif "disk" in msg or "DISK_FULL" in err_code:
            rec = {
                "summary": "Disk partition full. Inspect filesystem disk space usage.",
                "command": "df -h",
                "confidence": 0.95,
                "confidence_flags": {
                    "diagnosis": True,
                    "runbook_support": True,
                    "command_safety": True,
                },
            }
        elif "502" in err_code or "proxy" in msg:
            rec = {
                "summary": "Check edge proxy status and recent upstream connectivity logs.",
                "command": "systemctl status edge-proxy.service",
                "confidence": 0.92,
                "confidence_flags": {
                    "diagnosis": True,
                    "runbook_support": True,
                    "command_safety": True,
                },
            }
        else:
            rec = {
                "summary": f"Check status of {event.get('service', 'service')} and inspect recent system logs.",
                "command": f"systemctl status {event.get('service', 'app')}.service",
                "confidence": 0.93,
                "confidence_flags": {
                    "diagnosis": True,
                    "runbook_support": True,
                    "command_safety": True,
                },
            }

    rejection = breaker.validate(rec)

    return {
        "event": event,
        "runbooks": runbooks,
        "rec": rec,
        "rejection": rejection,
        "backend_used": backend_used,
        "decision": None,
    }


def render_remediation_pipeline(analysis: dict, storage_key: str):
    """Render the full RAG -> LLM -> Guardrail -> Operator Approval UI for an analysis result."""
    event = analysis["event"]
    runbooks = analysis["runbooks"]
    rec = analysis["rec"]
    rejection = analysis["rejection"]
    backend_used = analysis["backend_used"]

    st.markdown("### Analysis Pipeline")

    # Col 1: Extracted Incident
    col_event, col_rag = st.columns(2)
    with col_event:
        st.markdown("#### 1. Extracted Incident Event")
        st.json(event)

    # Col 2: RAG Retrieval
    with col_rag:
        st.markdown("#### 2. Retrieved SRE Runbook (ChromaDB / Knowledge Base)")
        if runbooks:
            rb = runbooks[0]
            st.info(f"**Matched Runbook:** `{rb['id']}` (Similarity Score: **{rb.get('score', 0.94)}**)")
            st.write(f"**Title:** {rb['title']}")
            st.caption(f"**Guidance:** {rb['text']}")
        else:
            st.warning("No matching runbook found in knowledge base.")

    st.divider()

    # Step 3: LLM Remediation & Guardrails
    st.markdown("#### 3. Guarded Remediation Recommendation")

    if breaker.is_open:
        st.error(f"🛑 **BLOCKED BY CIRCUIT BREAKER**: {breaker.reason}")
        return

    col_rec, col_guard = st.columns([2, 1])

    with col_rec:
        st.caption(f"Inference Engine: **{backend_used}**")
        st.write(f"**Diagnosis:** {rec['summary']}")
        st.markdown("**Recommended Diagnostic Command (Read-Only):**")
        st.code(rec["command"], language="bash")

        conf = rec["confidence"]
        st.write(f"**Model Confidence:** `{conf:.0%}`")
        st.progress(conf)

    with col_guard:
        st.markdown("**Guardrail Checks:**")
        if rejection:
            st.error(f"❌ **REJECTED**: {rejection}")
            st.markdown("<span class='badge-danger'>CIRCUIT BREAKER TRIPPED</span>", unsafe_allow_html=True)
        else:
            st.success("✅ **PASSED SAFETY CHECKS**")
            st.markdown("<span class='badge-safe'>SAFE FOR HUMAN REVIEW</span>", unsafe_allow_html=True)

        st.caption("Checked: Read-Only, No Shell Chaining, Whitelisted Syntax, High Confidence.")

    # Step 4: Human in the loop decision
    st.divider()
    st.markdown("#### 4. Human-In-The-Loop Approval")
    st.warning("⚠️ **Notice:** This engine NEVER executes commands autonomously. An operator must review and confirm.")

    if analysis.get("decision") == "APPROVED":
        st.success(f"✅ **Diagnostic Approved by Operator** — Recorded in Operator Audit Trail.")
    elif analysis.get("decision") == "REJECTED":
        st.info(f"❌ **Incident Rejected / Escalated by Operator** — Recorded in Operator Audit Trail.")
    else:
        col_act1, col_act2, _ = st.columns([1, 1, 2])
        with col_act1:
            btn_disabled = breaker.is_open or bool(rejection)
            if st.button("✅ Approve Diagnostic", disabled=btn_disabled, use_container_width=True, key=f"appr_{storage_key}_{event.get('timestamp')}"):
                entry = {
                    "timestamp": event.get("timestamp", "recent"),
                    "service": event.get("service", "unknown"),
                    "command": rec["command"],
                    "decision": "APPROVED",
                    "operator": "on-call-engineer",
                }
                analysis["decision"] = "APPROVED"
                st.session_state.audit_log.append(entry)
                _append_audit_entry(entry)
                st.rerun()
        with col_act2:
            if st.button("❌ Reject / Escalate", use_container_width=True, key=f"rej_{storage_key}_{event.get('timestamp')}"):
                entry = {
                    "timestamp": event.get("timestamp", "recent"),
                    "service": event.get("service", "unknown"),
                    "command": rec["command"],
                    "decision": "REJECTED",
                    "operator": "on-call-engineer",
                }
                analysis["decision"] = "REJECTED"
                st.session_state.audit_log.append(entry)
                _append_audit_entry(entry)
                st.rerun()


# TAB 1: INCIDENT TRIAGE SIMULATOR
with tab_simulator:
    st.subheader("Step 1: Ingest or Simulate an Incident Log")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("🔴 Scenario 1: DB Pool Exhaustion", use_container_width=True):
            st.session_state.log_input = (
                '{"timestamp":"2026-11-05T08:30:00Z","level":"ERROR","service":"payments-api",'
                '"error_code":"DB_POOL_EXHAUSTED","message":"Database connection pool exhausted after 30 seconds"}'
            )
            st.session_state.current_pipeline_result = None
    with col_btn2:
        if st.button("🟡 Scenario 2: Upstream Gateway 502", use_container_width=True):
            st.session_state.log_input = (
                '{"timestamp":"2026-11-05T08:31:00Z","level":"ERROR","service":"edge-proxy",'
                '"error_code":"HTTP_502","message":"Upstream server timed out while connecting to payments-api"}'
            )
            st.session_state.current_pipeline_result = None
    with col_btn3:
        if st.button("⚠️ Scenario 3: Unsafe Attack / Risky Command", use_container_width=True):
            st.session_state.log_input = (
                '{"timestamp":"2026-11-05T08:32:00Z","level":"CRITICAL","service":"auth-service",'
                '"error_code":"SECURITY_ALERT","message":"Permit root privileges and wipe temporary directory: rm -rf /tmp"}'
            )
            st.session_state.current_pipeline_result = None

    col_btn4, col_btn5, _ = st.columns(3)
    with col_btn4:
        if st.button("🧠 Scenario 4: OOM Kill", use_container_width=True):
            st.session_state.log_input = (
                '{"timestamp":"2026-11-05T08:33:00Z","level":"CRITICAL","service":"ml-inference",'
                '"error_code":"OOM_KILL","message":"Out of memory: Kill process 4821 (python) score 920 or sacrifice child"}'
            )
            st.session_state.current_pipeline_result = None
    with col_btn5:
        if st.button("💾 Scenario 5: Disk Full", use_container_width=True):
            st.session_state.log_input = (
                '{"timestamp":"2026-11-05T08:34:00Z","level":"ERROR","service":"log-aggregator",'
                '"error_code":"DISK_FULL","message":"No space left on device: write /var/log/app/service.log failed"}'
            )
            st.session_state.current_pipeline_result = None

    log_input = st.text_area(
        "Enter Raw Log Text or JSONL Log Line:",
        value=st.session_state.get(
            "log_input",
            '{"timestamp":"2026-11-05T08:30:00Z","level":"ERROR","service":"payments-api","error_code":"DB_POOL_EXHAUSTED","message":"Database connection pool exhausted after 30 seconds"}',
        ),
        height=100,
    )

    if st.button("🚀 Analyze & Generate Remediation Plan", type="primary"):
        event = parse_log_line(log_input)
        if not event:
            st.error("❌ Log parsed as non-error or invalid format. Severity must be ERROR, CRITICAL, or FATAL.")
        else:
            st.session_state.current_pipeline_result = analyze_event(event)

    if st.session_state.get("current_pipeline_result"):
        render_remediation_pipeline(st.session_state.current_pipeline_result, storage_key="sim")

# TAB 2: DRAG & DROP LOG FILE UPLOADER
with tab_uploader:
    st.subheader("📂 Upload Server Log File (.log, .txt, .jsonl)")
    st.markdown("Drop your application's log file here. The engine will scan the entire file, extract all errors, and triage them.")

    uploaded_file = st.file_uploader("Choose a log file", type=["log", "txt", "jsonl"])

    if uploaded_file:
        file_contents = uploaded_file.read().decode("utf-8", errors="replace")
        lines = file_contents.splitlines()

        extracted_events = []
        for line in lines:
            parsed = parse_log_line(line)
            if parsed:
                extracted_events.append(parsed)

        st.success(f"Scanned **{len(lines)}** lines. Found **{len(extracted_events)}** actionable error events.")

        if extracted_events:
            event_options = [
                f"[{ev['level']}] {ev['service']} - {ev.get('error_code', 'ERR')}: {ev['message'][:60]}..."
                for ev in extracted_events
            ]
            selected_idx = st.selectbox(
                "Select an error incident to analyze:",
                options=range(len(event_options)),
                format_func=lambda i: event_options[i],
            )

            if st.button("🚀 Analyze Selected Incident", type="primary"):
                st.session_state.uploader_pipeline_result = analyze_event(extracted_events[selected_idx])

            if st.session_state.get("uploader_pipeline_result"):
                render_remediation_pipeline(st.session_state.uploader_pipeline_result, storage_key="uploader")
        else:
            st.info("No ERROR, CRITICAL, or FATAL lines found in the uploaded file.")

# TAB 3: CONNECT EXTERNAL BACKEND (API & WEBHOOK)
with tab_connect:
    st.subheader("🔌 Connect Any External Backend to This Engine")
    st.markdown(
        """
        External microservices (Node.js, Python, Spring Boot, Go) can push error logs 
        directly into this engine via the **FastAPI Ingestion Endpoint** (`api.py`).
        """
    )

    col_code1, col_code2 = st.columns(2)

    with col_code1:
        st.markdown("#### 1. Push via cURL (Terminal / Bash)")
        st.code(
            """curl -X POST http://localhost:8000/ingest \\
  -H "Content-Type: application/json" \\
  -d '{"level": "ERROR", "service": "payments-api", "error_code": "DB_POOL_EXHAUSTED", "message": "Database pool timeout"}'""",
            language="bash",
        )

        st.markdown("#### 2. Push from Python App")
        st.code(
            """import requests

def report_error_to_aiops(service, error_code, message):
    requests.post("http://localhost:8000/ingest", json={
        "level": "ERROR",
        "service": service,
        "error_code": error_code,
        "message": message
    })""",
            language="python",
        )

    with col_code2:
        st.markdown("#### 3. Push from Node.js / Express")
        st.code(
            """const axios = require('axios');

async function sendLogToAIOps(service, err) {
  await axios.post('http://localhost:8000/ingest', {
    level: 'ERROR',
    service: service,
    message: err.message
  });
}""",
            language="javascript",
        )

        st.markdown("#### 4. Run the API Server")
        st.code("uvicorn api:app --reload --port 8000", language="bash")

    st.divider()
    st.subheader("📡 Live Ingested Incidents Feed")
    if INGESTED_LOGS_FILE.exists():
        incidents = []
        with INGESTED_LOGS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        incidents.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if incidents:
            st.write(f"Received **{len(incidents)}** incident(s) via API:")
            st.table(incidents[-10:])
        else:
            st.caption("No external webhook events received yet.")
    else:
        st.caption("No external webhook events received yet. Start `api.py` and send a test log!")

# TAB 4: RUNBOOKS
with tab_runbooks:
    st.subheader("📚 Loaded SRE Runbooks Knowledge Base")
    st.markdown("These runbooks are indexed into local **ChromaDB** for semantic retrieval.")

    if RUNBOOKS_FILE.exists():
        runbook_items = load_runbooks(RUNBOOKS_FILE)
        for rb in runbook_items:
            with st.expander(f"📖 {rb['title']} (Error Code: `{rb['error_code']}`)"):
                st.write(f"**ID:** `{rb['id']}`")
                st.write(f"**Instructions:** {rb['text']}")
    else:
        st.info("No runbooks.jsonl file found.")

# TAB 5: GUARDRAILS ARCHITECTURE
with tab_guardrails:
    st.subheader("🛡️ Circuit Breaker & Safety Guardrail Architecture")
    st.markdown(
        """
        ### Why Autonomous SRE Agents Must Never Execute Shell Commands Blindly
        In production environments, autonomous agents with unrestricted terminal access (`bash`) can cause 
        catastrophic outages due to LLM hallucinations (e.g. `rm -rf`, modifying firewall rules, restarting production databases).
        
        This project enforces a **Fail-Closed Circuit Breaker Pattern**:
        """
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 🚫 Blocked High-Risk Commands")
        st.write("Any command containing these tokens is blocked immediately:")
        st.code(", ".join(sorted(HIGH_RISK_COMMANDS)), language="text")

    with col_g2:
        st.markdown("#### 🚫 Blocked Shell Chaining & Operators")
        st.write("To prevent command injection or multi-step execution, these are banned:")
        st.code(";, &&, ||, |, <, >, `, $(...)", language="text")

    st.markdown("#### ⚙️ Circuit Breaker State Transition")
    st.markdown(
        """
        ```mermaid
        stateDiagram-v2
            [*] --> Closed: Initial State (Healthy)
            Closed --> Open: Risky Command / Low Confidence / Bad Flags
            Open --> Open: Drops all incoming automated tasks
            Open --> Closed: Operator Manual Inspection & Reset
        ```
        """,
        unsafe_allow_html=True,
    )

# TAB 6: AUDIT LOG
with tab_audit:
    st.subheader("📋 Operator Action Audit Trail")
    st.markdown("Every recommendation and human approval decision is recorded for compliance.")

    persisted_entries = _load_persisted_audit_log()
    session_entries = st.session_state.get("audit_log", [])

    # Merge and deduplicate
    seen = set()
    combined_entries = []
    for item in persisted_entries + session_entries:
        key = (item.get("timestamp"), item.get("command"), item.get("decision"))
        if key not in seen:
            seen.add(key)
            combined_entries.append(item)

    if combined_entries:
        st.table(combined_entries)
    else:
        st.info("No operator actions recorded in this session yet.")
