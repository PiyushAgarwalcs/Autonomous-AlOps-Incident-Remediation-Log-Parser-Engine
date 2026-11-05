import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from aioops_engine import CircuitBreaker, OllamaClient, parse_log_line
from core.parser import follow_file
from core.vector_store import ChromaRunbooks, cosine_similarity
from pathlib import Path


class ParseLogLineTests(unittest.TestCase):
    def test_parses_structured_error_event(self):
        event = parse_log_line(
            '{"timestamp":"2026-11-03T08:30:00Z","level":"ERROR",'
            '"service":"payments","error_code":"DB_POOL","message":"pool exhausted"}'
        )
        self.assertEqual(event["service"], "payments")
        self.assertEqual(event["error_code"], "DB_POOL")

    def test_ignores_non_errors_and_parses_raw_errors(self):
        self.assertIsNone(parse_log_line('{"level":"INFO","message":"ready"}'))
        self.assertIsNone(parse_log_line("server started successfully"))
        event = parse_log_line("2026-11-03 CRITICAL payments: connection refused")
        self.assertEqual(event["level"], "CRITICAL")


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        self.breaker = CircuitBreaker(minimum_confidence=0.75)
        self.recommendation = {
            "summary": "Check the service state.",
            "command": "systemctl status payments-api.service",
            "confidence": 0.9,
            "confidence_flags": {
                "diagnosis": True,
                "runbook_support": True,
                "command_safety": True,
            },
        }

    def test_accepts_safe_recommendation_without_executing_it(self):
        self.assertIsNone(self.breaker.validate(self.recommendation))

    def test_blocks_high_risk_command_and_opens_breaker(self):
        self.recommendation["command"] = "chmod 777 /tmp/service"
        self.assertIn("High-risk", self.breaker.validate(self.recommendation))
        self.assertTrue(self.breaker.is_open)
        self.assertIsNotNone(self.breaker.validate({"command": "systemctl status x"}))

    def test_blocks_risky_command_wrapped_in_shell_argument(self):
        self.recommendation["command"] = 'bash -c "rm -rf /tmp/service"'
        self.assertIn("High-risk", self.breaker.validate(self.recommendation))

    def test_low_confidence_opens_breaker(self):
        self.recommendation["confidence"] = 0.4
        self.assertIn("confidence", self.breaker.validate(self.recommendation))
        self.assertTrue(self.breaker.is_open)

    def test_false_confidence_flag_opens_breaker(self):
        self.recommendation["confidence_flags"]["runbook_support"] = False
        self.assertIn("confidence", self.breaker.validate(self.recommendation))

    def test_rejects_non_finite_confidence(self):
        self.recommendation["confidence"] = float("nan")
        self.assertIn("schema", self.breaker.validate(self.recommendation))

    def test_rejects_shell_chaining(self):
        self.recommendation["command"] = "systemctl status x; systemctl restart x"
        self.assertIn("Shell operators", self.breaker.validate(self.recommendation))

    def test_reset_circuit_breaker(self):
        self.breaker.trip("Manual trip for testing")
        self.assertTrue(self.breaker.is_open)
        self.breaker.reset()
        self.assertFalse(self.breaker.is_open)
        self.assertEqual(self.breaker.reason, "")
        self.assertIsNone(self.breaker.validate(self.recommendation))


class OllamaLocalOnlyTests(unittest.TestCase):
    def test_rejects_remote_ollama_host(self):
        with self.assertRaises(ValueError):
            OllamaClient("https://example.com", "qwen2.5:7b", "nomic-embed-text")


class VectorStoreTests(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_chroma_runbooks_search_with_mock_ollama(self):
        store = ChromaRunbooks(persist_dir="./test_chroma_db", collection_name="test_runbooks")
        mock_ollama = MagicMock()
        mock_ollama.embed.side_effect = lambda text: [1.0 if "pool" in text.lower() else 0.0, 0.5, 0.1]

        runbooks = [
            {
                "id": "rb-1",
                "error_code": "DB_POOL_EXHAUSTED",
                "title": "Database connection pool exhausted",
                "text": "Check payments-api database pool limits.",
            }
        ]
        count = store.upsert(runbooks, mock_ollama)
        self.assertEqual(count, 1)

        results = store.search("DB connection pool is empty", mock_ollama, top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "rb-1")


class FollowFileTests(unittest.TestCase):
    def test_reads_existing_lines(self):
        """follow_file should yield lines already in the file when from_end=False."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as fh:
            fh.write('{"level":"ERROR","message":"db down"}\n')
            fh.write('{"level":"INFO","message":"ok"}\n')
            path = Path(fh.name)

        gen = follow_file(path, poll_interval=0.05, from_end=False)
        collected = []
        try:
            # Read exactly 2 pre-existing lines then close the generator
            collected.append(next(gen))
            collected.append(next(gen))
        finally:
            gen.close()  # Triggers the finally block → closes the file handle
            time.sleep(0.05)
            try:
                os.unlink(path)
            except PermissionError:
                pass

        self.assertEqual(len(collected), 2)
        self.assertIn("db down", collected[0])

    def test_from_end_skips_existing_content(self):
        """follow_file with from_end=True should not yield pre-existing lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as fh:
            fh.write('{"level":"ERROR","message":"old error"}\n')
            path = Path(fh.name)

        result_lines: list[str] = []
        result_error: list[Exception] = []

        def _tail():
            gen = follow_file(path, poll_interval=0.05, from_end=True)
            try:
                # Block until one line is available (the appended line)
                result_lines.append(next(gen))
            except Exception as exc:
                result_error.append(exc)
            finally:
                gen.close()

        tail_thread = threading.Thread(target=_tail, daemon=True)
        tail_thread.start()

        # Wait briefly so the tailer positions at EOF, then append a new line
        time.sleep(0.2)
        with path.open("a", encoding="utf-8") as ap:
            ap.write('{"level":"ERROR","message":"new error"}\n')

        tail_thread.join(timeout=2.0)
        time.sleep(0.05)
        try:
            os.unlink(path)
        except PermissionError:
            pass

        if result_error:
            self.fail(f"follow_file raised an unexpected error: {result_error[0]}")

        self.assertTrue(any("new error" in line for line in result_lines),
                        "Should have read the newly appended line.")
        self.assertFalse(any("old error" in line for line in result_lines),
                         "Should have skipped the pre-existing line.")


if __name__ == "__main__":
    unittest.main()
