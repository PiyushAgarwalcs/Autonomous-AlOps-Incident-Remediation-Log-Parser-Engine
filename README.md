# AIOps Log Parser & Remediation Engine

A lightweight Python CLI that follows a local JSON Lines or plain-text server log, retrieves matching SRE runbooks from Pinecone, and asks a text-only Ollama model for a structured remediation recommendation. It has no desktop control and never executes commands.

## Requirements

- Python 3.10+
- Ollama running on `localhost:11434`
- `qwen2.5:7b` for text generation and `nomic-embed-text` for local embeddings
- A Pinecone index configured for **768 dimensions** and cosine similarity

Inference, embeddings, parsing, and command validation run locally. Pinecone remains a hosted dependency because this project retains Pinecone for vector retrieval; it must be replaced with a local vector store for a fully air-gapped deployment.

## Install and configure

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
$env:PINECONE_API_KEY = "your-pinecone-api-key"
$env:PINECONE_INDEX = "your-768-dimension-index"
```

Start Ollama (for example, `ollama serve`) and make sure the Pinecone index already exists.

## Load runbooks and monitor logs

Runbook documents are JSONL records in `runbooks.jsonl`. Index them once:

```powershell
python aioops_engine.py --sync-runbooks
```

Process the included sample file once:

```powershell
python aioops_engine.py --log-file samples/server.jsonl --once
```

Follow a live log, processing new lines as they arrive:

```powershell
python aioops_engine.py --log-file C:\path\to\server.log
```

Add `--from-end` to ignore existing lines and wait only for new log entries. Plain-text lines are considered only when they contain `ERROR`, `CRITICAL`, or `FATAL`; JSON Lines records use a `level` field and an error `message`.

The engine emits JSON status records to stdout. A recommendation is displayed only when its confidence meets the configured threshold, all confidence flags are true, and the command passes the safety checks. Risky or low-confidence output opens the circuit breaker and drops the task; restart the process after investigating. The command is always a recommendation for human review and is never run by this program.

## Input formats

Structured JSON Lines (`samples/server.jsonl`):

```json
{"timestamp":"2026-11-03T08:30:00Z","level":"ERROR","service":"payments-api","error_code":"DB_POOL_EXHAUSTED","message":"Database connection pool exhausted after 30 seconds"}
```

Raw text is also accepted:

```text
2026-11-03 08:31:00 CRITICAL payments-api: database connection refused
```

Runbooks (`runbooks.jsonl`) use `id`, `error_code`, `title`, and `text` fields. Change models, Ollama URL, and confidence threshold with `--chat-model`, `--embedding-model`, `--ollama-url`, and `--min-confidence`.

## Tests

```powershell
python -m unittest discover -s tests
```

# Autonomous-AlOps-Incident-Remediation-Log-Parser-Engine
