"""Vector store integration using local ChromaDB for SRE runbook retrieval."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import OllamaClient


def load_runbooks(path: Path) -> list[dict[str, str]]:
    """Load and validate runbook records from a JSONL file."""
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


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ChromaRunbooks:
    """Local, air-gapped vector store for SRE runbooks using ChromaDB (with local fallback)."""

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "sre_runbooks") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._fallback_store: list[dict[str, Any]] = []
        self._chroma_client = None
        self._collection = None

        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Fallback to local in-memory/JSON storage if chromadb native bindings unavailable
            self._chroma_client = None
            self._collection = None

    @property
    def is_using_chroma(self) -> bool:
        return self._collection is not None

    def upsert(self, runbooks: list[dict[str, str]], ollama: OllamaClient) -> int:
        """Embed and upsert runbooks into the vector collection."""
        if not runbooks:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for runbook in runbooks:
            searchable_text = f"{runbook['error_code']} {runbook['title']}\n{runbook['text']}"
            vector = ollama.embed(searchable_text)
            ids.append(runbook["id"])
            embeddings.append(vector)
            documents.append(searchable_text)
            metadatas.append({
                "error_code": runbook["error_code"],
                "title": runbook["title"],
                "text": runbook["text"],
            })

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            # Fallback storage
            for r_id, emb, meta in zip(ids, embeddings, metadatas):
                self._fallback_store = [
                    item for item in self._fallback_store if item["id"] != r_id
                ]
                self._fallback_store.append({
                    "id": r_id,
                    "embedding": emb,
                    "metadata": meta,
                })

        return len(runbooks)

    def search(self, query: str, ollama: OllamaClient, top_k: int = 3) -> list[dict[str, Any]]:
        """Search runbooks semantically using query text."""
        query_vector = ollama.embed(query)

        if self._collection is not None:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, count),
                include=["metadatas", "distances"],
            )
            matches = []
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc_id, meta, dist in zip(ids, metadatas, distances):
                # Chroma cosine distance is 1 - similarity
                score = round(1.0 - float(dist), 4) if dist is not None else 1.0
                matches.append({
                    "id": doc_id,
                    "score": score,
                    "error_code": meta.get("error_code", ""),
                    "title": meta.get("title", ""),
                    "text": meta.get("text", ""),
                })
            return matches

        # Fallback search
        if not self._fallback_store:
            return []

        scored = []
        for item in self._fallback_store:
            score = cosine_similarity(query_vector, item["embedding"])
            scored.append({
                "id": item["id"],
                "score": round(score, 4),
                "error_code": item["metadata"].get("error_code", ""),
                "title": item["metadata"].get("title", ""),
                "text": item["metadata"].get("text", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

